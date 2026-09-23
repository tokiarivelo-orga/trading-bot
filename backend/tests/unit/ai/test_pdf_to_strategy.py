"""PDF -> StrategySpec -> code pipeline (§8.1): human review is never
skippable — edits reset review, approval gates code generation, and
generated code always goes through the sandbox before anything is written
to `strategies/generated/`."""

from __future__ import annotations

import json

import fitz
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.ai.adapters.repository import DraftRepository
from src.ai.application.pdf_to_strategy import InvalidDraftStateError, PdfToStrategyService
from src.ai.domain.models import (
    AnnotationType,
    DraftStatus,
    ExtractedStrategySpec,
    IndicatorSpec,
    IndicatorType,
    PriceLevelAnnotation,
)
from src.ai.ports.llm import LLMCallError
from src.market_data.adapters import orm as market_data_orm  # noqa: F401 — registers candles table
from src.shared.db.base import Base
from src.strategies.adapters.repository import StrategyVersionRepository
from src.strategies.application.versioning import StrategyVersionService
from src.strategies.domain.versioning import VersionStatus
from src.strategies.registry import StrategyRegistry


def _code_for(name: str) -> str:
    class_name = "".join(w.capitalize() for w in name.split("_"))
    return f"""
from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec


class {class_name}:
    def __init__(self):
        self.spec = StrategySpec(
            name="{name}", version=1, symbols=("XAUUSD",), entry_timeframe="M5",
            confirmation_timeframes=("H1",), params={{}},
        )

    def evaluate(self, ctx: MarketContext):
        return None
"""


def _fake_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Buy the M5 EMA200 pullback in an uptrend.")
    data = doc.tobytes()
    doc.close()
    return data


EXTRACTED_SPEC_JSON = json.dumps(
    {
        "name": "gold_ema_pullback",
        "symbols": ["XAUUSD"],
        "entry_timeframe": "M5",
        "confirmation_timeframes": ["H1"],
        "indicators": [{"type": "ema", "period": 200, "label": "EMA200"}],
        "unrecognized_indicators": ["Ichimoku Cloud"],
        "price_levels": [{"type": "resistance", "price": 2050.0, "label": "resistance at 2050"}],
        "chart_notes": ["Fibonacci retracement on the swing points"],
        "entry_rules": "Buy when price pulls back to EMA200 in an uptrend.",
        "exit_rules": "SL below recent swing low, TP at 2R.",
        "risk_notes": "Risk 0.5% per trade.",
        "params": {"ema_period": 200},
    }
)


class FakeExtractionLLM:
    async def complete(self, message, *, max_tokens=4096):
        return EXTRACTED_SPEC_JSON


class FakeCodeGenLLM:
    def __init__(self, code: str) -> None:
        self.code = code

    async def complete(self, message, *, max_tokens=4096):
        return f"```python\n{self.code}\n```"  # exercise fence-stripping too


class InvalidCodeGenLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, message, *, max_tokens=4096):
        self.calls += 1
        return "import os\nx = 1\n"


class RetryThenValidCodeGenLLM:
    """First completion fails the sandbox (forbidden import); the retry
    prompt (fed the sandbox's error) gets a valid one — mirrors an LLM
    self-correcting once shown what it broke."""

    def __init__(self, bad_code: str, good_code: str) -> None:
        self.bad_code = bad_code
        self.good_code = good_code
        self.calls = 0

    async def complete(self, message, *, max_tokens=4096):
        self.calls += 1
        code = self.bad_code if self.calls == 1 else self.good_code
        return f"```python\n{code}\n```"


class FailingCodeGenLLM:
    """Simulates a provider call that fails outright (e.g. the claude_code
    adapter's subprocess timing out) rather than returning bad code — the
    failure must propagate, not get swallowed as a sandbox rejection."""

    async def complete(self, message, *, max_tokens=4096):
        raise LLMCallError("claude code call exceeded 480s timeout and was killed")


class FakeRouter:
    def __init__(self, extraction_llm, codegen_llm) -> None:
        self._map = {"pdf_extraction": extraction_llm, "code_generation": codegen_llm}

    def for_task(self, task: str):
        return self._map[task]


def _make_service(tmp_path, codegen_llm):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    strategy_versions = StrategyVersionService(
        StrategyVersionRepository(session_factory), StrategyRegistry(), generated_dir
    )
    draft_repository = DraftRepository(session_factory)
    router = FakeRouter(FakeExtractionLLM(), codegen_llm)

    candles_engine = create_engine(f"sqlite:///{tmp_path}/candles.db")
    Base.metadata.create_all(candles_engine)  # empty candles table -> NoHistoryError, handled

    return PdfToStrategyService(
        draft_repository,
        strategy_versions,
        router,
        backtest_database_url=f"sqlite:///{tmp_path}/candles.db",
    )


@pytest.fixture
def service(tmp_path):
    return _make_service(tmp_path, FakeCodeGenLLM(_code_for("gold_ema_pullback")))


async def test_create_draft_from_pdf_extracts_spec(service):
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    assert draft.status == DraftStatus.PENDING_REVIEW
    assert draft.extracted_spec.name == "gold_ema_pullback"
    assert draft.edited_spec is None
    assert draft.effective_spec == draft.extracted_spec


async def test_create_draft_from_pdf_with_symbol_overrides_extraction(service):
    # The LLM's extraction (EXTRACTED_SPEC_JSON) always guesses XAUUSD since
    # the fake PDF text never names an instrument — the caller's symbol (the
    # one active on the chart) must win in effective_spec, while the raw
    # extraction is kept untouched for audit.
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes(), symbol="EURUSD")
    assert draft.extracted_spec.symbols == ("XAUUSD",)
    assert draft.edited_spec is not None
    assert draft.edited_spec.symbols == ("EURUSD",)
    assert draft.effective_spec.symbols == ("EURUSD",)
    # Only symbols is overridden — the rest of the extraction carries through.
    assert draft.effective_spec.name == draft.extracted_spec.name
    assert draft.effective_spec.indicators == draft.extracted_spec.indicators


async def test_create_draft_from_pdf_extracts_structured_indicators_and_levels(service):
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    spec = draft.extracted_spec
    assert spec.indicators == (IndicatorSpec(type=IndicatorType.EMA, period=200, label="EMA200"),)
    assert spec.unrecognized_indicators == ("Ichimoku Cloud",)
    assert spec.price_levels == (
        PriceLevelAnnotation(
            type=AnnotationType.RESISTANCE, price=2050.0, label="resistance at 2050"
        ),
    )
    assert spec.chart_notes == ("Fibonacci retracement on the swing points",)


async def test_create_draft_from_spec_skips_llm_and_uses_spec_as_is(service):
    spec = ExtractedStrategySpec.from_dict(json.loads(EXTRACTED_SPEC_JSON))

    draft = await service.create_draft_from_spec(spec, filename="my_upload.json")

    assert draft.status == DraftStatus.PENDING_REVIEW
    assert draft.source_filename == "my_upload.json"
    assert draft.extracted_spec == spec
    assert draft.edited_spec is None
    assert draft.effective_spec == spec


async def test_create_draft_from_spec_with_symbol_overrides_spec(service):
    spec = ExtractedStrategySpec.from_dict(json.loads(EXTRACTED_SPEC_JSON))

    draft = await service.create_draft_from_spec(spec, symbol="EURUSD")

    assert draft.extracted_spec.symbols == ("XAUUSD",)  # untouched, for audit
    assert draft.edited_spec is not None
    assert draft.edited_spec.symbols == ("EURUSD",)
    assert draft.effective_spec.symbols == ("EURUSD",)


async def test_update_draft_spec_keeps_original_and_resets_review(service):
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    edited = ExtractedStrategySpec.from_dict({**draft.extracted_spec.to_dict(), "name": "renamed"})

    updated = await service.update_draft_spec(draft.id, edited)
    assert updated.status == DraftStatus.PENDING_REVIEW
    assert updated.edited_spec.name == "renamed"
    assert updated.extracted_spec.name == "gold_ema_pullback"  # untouched
    assert updated.effective_spec.name == "renamed"


async def test_approve_requires_pending_review(service):
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    await service.approve_draft(draft.id)
    with pytest.raises(InvalidDraftStateError):
        await service.approve_draft(draft.id)


async def test_generate_code_requires_approval(service):
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    with pytest.raises(InvalidDraftStateError):
        await service.generate_code(draft.id)


async def test_generate_code_success_creates_validated_version(service):
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    await service.approve_draft(draft.id)

    result = await service.generate_code(draft.id)

    assert result.is_valid
    assert result.sandbox_errors == ()
    assert result.version_id is not None
    # No candle history in the isolated test DB -> auto-backtest cleanly skips.
    assert result.backtest_report_id is None

    updated_draft = await service.get_draft(draft.id)
    assert updated_draft.status == DraftStatus.CODE_GENERATED

    version = service._strategy_versions.get_version(result.version_id)
    assert version.status == VersionStatus.VALIDATED
    assert version.source.value == "ai_generated"
    assert version.draft_id == draft.id


async def test_generate_code_rejects_invalid_code_without_crashing(tmp_path):
    llm = InvalidCodeGenLLM()
    service = _make_service(tmp_path, llm)
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    await service.approve_draft(draft.id)

    result = await service.generate_code(draft.id)

    assert not result.is_valid
    assert result.version_id is None
    assert any("os" in e for e in result.sandbox_errors)
    # 1 initial completion + 2 retries (MAX_ATTEMPTS=3 sandbox checks total),
    # then gives up rather than looping forever against an LLM that never
    # fixes the problem.
    assert llm.calls == 3

    # Draft stays approved (not code_generated) so the user can retry.
    updated_draft = await service.get_draft(draft.id)
    assert updated_draft.status == DraftStatus.APPROVED


async def test_generate_code_retries_after_sandbox_rejection_then_succeeds(tmp_path):
    llm = RetryThenValidCodeGenLLM(
        bad_code="import os\nx = 1\n", good_code=_code_for("gold_ema_pullback")
    )
    service = _make_service(tmp_path, llm)
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    await service.approve_draft(draft.id)

    result = await service.generate_code(draft.id)

    assert result.is_valid
    assert result.sandbox_errors == ()
    assert result.version_id is not None
    assert llm.calls == 2

    version = service._strategy_versions.get_version(result.version_id)
    assert version.status == VersionStatus.VALIDATED


async def test_generate_code_propagates_llm_call_error(tmp_path):
    service = _make_service(tmp_path, FailingCodeGenLLM())
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    await service.approve_draft(draft.id)

    with pytest.raises(LLMCallError):
        await service.generate_code(draft.id)

    # Draft stays approved (not code_generated) so the user can retry.
    updated_draft = await service.get_draft(draft.id)
    assert updated_draft.status == DraftStatus.APPROVED


async def test_reject_draft(service):
    draft = await service.create_draft_from_pdf("method.pdf", _fake_pdf_bytes())
    rejected = await service.reject_draft(draft.id)
    assert rejected.status == DraftStatus.REJECTED


async def test_create_draft_from_text_extracts_spec(service):
    draft = await service.create_draft_from_text(
        "Buy when price pulls back to the 200 EMA on H1 with RSI below 40."
    )
    assert draft.source_filename == "(typed prompt)"
    assert draft.status == DraftStatus.PENDING_REVIEW
    assert draft.extracted_spec.name == "gold_ema_pullback"
    assert draft.edited_spec is None
    assert draft.effective_spec == draft.extracted_spec


async def test_create_draft_from_text_with_symbol_overrides_extraction(service):
    draft = await service.create_draft_from_text("Buy the pullback in an uptrend.", symbol="EURUSD")
    assert draft.extracted_spec.symbols == ("XAUUSD",)
    assert draft.edited_spec is not None
    assert draft.edited_spec.symbols == ("EURUSD",)
    assert draft.effective_spec.symbols == ("EURUSD",)


async def test_create_draft_from_text_can_be_approved_and_generate_code(service):
    draft = await service.create_draft_from_text("Buy the pullback in an uptrend.")
    await service.approve_draft(draft.id)

    result = await service.generate_code(draft.id)

    assert result.is_valid
    assert result.version_id is not None
