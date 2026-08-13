"""Wire models for strategy versioning/activation (§6.5, §8.1). Mirrors
`strategies/domain/versioning.py`; the domain stays framework-free.

`StrategySpecSnapshotOut` intentionally duplicates the shape of
`ai/api/schemas.py: ExtractedStrategySpecSchema` rather than importing it —
api/schemas.py mirrors this module's own domain, it never reaches into
another module's internals (see CLAUDE.md "Architecture").
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.strategies.domain.versioning import CodeSource, StrategyVersion, VersionStatus

# Recognizes the old pre-structured indicator shape ("EMA200", "RSI(14)") so
# `StrategyVersion.spec` rows written before indicators became structured
# objects still deserialize instead of 500ing GET /strategies/versions. Kept
# as an independent copy of `ai/domain/models.py`'s equivalent regex — this
# module never imports from `ai/` (see module docstring above).
_LEGACY_INDICATOR_TOKEN_RE = re.compile(r"^(EMA|SMA|RSI)\s*\(?(\d+)\)?$", re.IGNORECASE)


class IndicatorSpecOut(BaseModel):
    """One indicator recognized into a plottable family (ema/sma/rsi/macd/
    bollinger) — mirrors `ai/api/schemas.py: IndicatorSpecSchema`."""

    type: str = Field(description="One of: ema, sma, rsi, macd, bollinger.")
    period: int = Field(
        description="Primary lookback — EMA/SMA/RSI span, Bollinger's SMA period, or MACD's "
        "fast period."
    )
    label: str = Field(description="The indicator as written in the source text, e.g. 'EMA200'.")
    source: str = Field(default="close", description="Candle field the indicator is computed on.")
    params: dict[str, float] = Field(
        default_factory=dict,
        description="Family-specific extra knobs — macd: {slow, signal}; bollinger: {std_dev}.",
    )


class PriceLevelAnnotationOut(BaseModel):
    """An explicit numeric price level the source text states outright —
    mirrors `ai/api/schemas.py: PriceLevelAnnotationSchema`."""

    type: str = Field(description="One of: support, resistance, level.")
    price: float = Field(description="The literal price level from the text.")
    label: str = Field(description="The level as written in the source text.")


class StrategySpecSnapshotOut(BaseModel):
    name: str = Field(description="Short snake_case slug, e.g. 'gold_ema_pullback'.")
    symbols: list[str] = Field(description="Symbols this method applies to.")
    entry_timeframe: str = Field(description="Entry timeframe — always 'M5' for this project.")
    confirmation_timeframes: list[str] = Field(description="Higher timeframes used to confirm.")
    indicators: list[IndicatorSpecOut] = Field(
        description="Indicators recognized into one of the 5 plottable families."
    )
    entry_rules: str = Field(description="Plain-English entry logic.")
    exit_rules: str = Field(description="Plain-English exit logic.")
    risk_notes: str = Field(description="Informational only — real caps live in risk.yaml.")
    params: dict[str, Any] = Field(default_factory=dict, description="Numeric parameters.")
    unrecognized_indicators: list[str] = Field(
        default_factory=list,
        description="Indicator names that don't map onto one of the 5 plottable families — "
        "display/audit only, never rendered on the chart.",
    )
    price_levels: list[PriceLevelAnnotationOut] = Field(
        default_factory=list,
        description="Explicit numeric support/resistance/pivot levels — rendered as locked "
        "horizontal lines on the chart.",
    )
    chart_notes: list[str] = Field(
        default_factory=list,
        description="Other charting/drawing-tool mentions with no explicit number attached — "
        "informational only, never rendered as geometry.",
    )


def _coerce_unrecognized_indicators(raw_entries: list[Any]) -> list[str]:
    """`unrecognized_indicators` must be `list[str]`, but some persisted specs
    hold `{"name": ..., "description": ...}` objects instead (hand-inserted
    versions, or a future extraction bug) — stringify those rather than
    500ing the whole versions listing on one bad row."""
    coerced: list[str] = []
    for entry in raw_entries:
        if isinstance(entry, str):
            coerced.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            description = entry.get("description")
            coerced.append(f"{entry['name']}: {description}" if description else entry["name"])
        else:
            coerced.append(str(entry))
    return coerced


def _coerce_legacy_spec_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a `StrategyVersion.spec` dict written before indicators were
    structured (plain `indicators: list[str]`, no `unrecognized_indicators`/
    `price_levels`/`chart_notes`) so it still validates against the current
    `StrategySpecSnapshotOut`. Specs already in the new shape pass through
    unchanged (each `indicators` entry is already a dict)."""
    indicators = raw.get("indicators", [])
    if indicators and all(isinstance(entry, str) for entry in indicators):
        parsed: list[dict[str, Any]] = []
        unrecognized: list[Any] = list(raw.get("unrecognized_indicators", []))
        for token in indicators:
            match = _LEGACY_INDICATOR_TOKEN_RE.match(str(token).strip())
            if match:
                family, period = match.group(1).lower(), int(match.group(2))
                parsed.append({"type": family, "period": period, "label": token})
            else:
                unrecognized.append(str(token))
        raw = {**raw, "indicators": parsed, "unrecognized_indicators": unrecognized}
    return {
        **raw,
        "unrecognized_indicators": _coerce_unrecognized_indicators(
            raw.get("unrecognized_indicators", [])
        ),
    }


class StrategyVersionOut(BaseModel):
    id: str = Field(description="Version id, used in every other /strategies/versions endpoint.")
    name: str = Field(description="Strategy family name — versions of the same strategy share it.")
    version: int = Field(description="1-based version number within this name.")
    file_path: str = Field(description="Path under backend/ where the source file lives.")
    code_hash: str = Field(description="SHA-256 of the source, for change detection/audit.")
    source: CodeSource = Field(
        description="'ai_generated' (via PDF pipeline), 'ai_refined' (via the 10-trade "
        "self-refinement loop), or 'manual'."
    )
    status: VersionStatus = Field(
        description="'validated' (passed the sandbox, not live), 'active' (registered in the "
        "StrategyRegistry and tradeable), or 'archived' (was active, superseded)."
    )
    paused: bool = Field(
        description="Only meaningful while status is 'active': true if the version is "
        "suspended from live evaluation via POST .../pause without being deactivated. "
        "Distinct from the engine-wide kill switch, which pauses every strategy at once."
    )
    created_at: int = Field(description="Epoch seconds UTC.")
    parent_version_id: str | None = Field(
        description="The version this one supersedes, if any — the rollback/diff chain."
    )
    draft_id: str | None = Field(
        description="The AI draft this came from, if source is ai_generated."
    )
    spec: StrategySpecSnapshotOut | None = Field(
        description="The StrategySpec this version was generated from, if source is ai_generated."
    )
    backtest_report_id: str | None = Field(
        description="Id of the backtest run when this version was generated (GET "
        "/backtest/reports/{id}), if one was run."
    )

    @staticmethod
    def from_domain(version: StrategyVersion) -> StrategyVersionOut:
        return StrategyVersionOut(
            id=version.id,
            name=version.name,
            version=version.version,
            file_path=version.file_path,
            code_hash=version.code_hash,
            source=version.source,
            status=version.status,
            paused=version.paused,
            created_at=int(version.created_at.timestamp()),
            parent_version_id=version.parent_version_id,
            draft_id=version.draft_id,
            spec=(
                StrategySpecSnapshotOut(**_coerce_legacy_spec_dict(version.spec))
                if version.spec
                else None
            ),
            backtest_report_id=version.backtest_report_id,
        )


class UpdateVersionSpecRequest(StrategySpecSnapshotOut):
    """Full replacement for a version's spec snapshot — same shape as
    `StrategySpecSnapshotOut`, submitted back after the trader hand-edits it
    on the version detail page rather than derived from code or AI
    extraction."""


class DuplicateVersionRequest(BaseModel):
    name: str = Field(
        description="New strategy family name for the duplicate — must not already be in "
        "use by another family, or the request is rejected."
    )
    symbols: list[str] | None = Field(
        default=None,
        description="Optional symbol list to retarget the duplicate to. Rewrites the "
        "`StrategySpec(symbols=...)` literal in the cloned source and re-validates it in "
        "the sandbox before saving; the request fails if no such literal can be found. "
        "Leave unset to duplicate with the same symbols as the source version. This never "
        "edits configs/app.yaml — the engine won't trade a new symbol live until it's applied "
        "to that symbol via PUT /skills/normal/{symbol} (see the `skills` tag), which is the "
        "one deliberate action that activates it.",
    )


class RenameVersionRequest(BaseModel):
    name: str = Field(
        description="New display name for this version's strategy family — applies to "
        "every version that currently shares the family's name, not just this one."
    )


class EditVersionCodeRequest(BaseModel):
    code: str = Field(
        description="Full replacement Python source for this strategy. Re-validated in the "
        "sandbox before saving; nothing is written if it fails."
    )
    new_name: str | None = Field(
        default=None,
        description="Leave unset to save as the next version of this version's own strategy "
        "family (the usual case). Set to a different, not-yet-used name to fork the edit into "
        "a brand-new strategy family at version 1 instead — the 'duplicate' save destination, "
        "for trying a change without touching the original. Rejected with 409 if the name is "
        "already in use by another family.",
    )


class StrategyVersionDetailOut(StrategyVersionOut):
    code: str = Field(description="The full generated Python source for this version.")

    @staticmethod
    def from_domain_with_code(version: StrategyVersion, code: str) -> StrategyVersionDetailOut:
        summary = StrategyVersionOut.from_domain(version).model_dump()
        return StrategyVersionDetailOut(**summary, code=code)


# ---------------------------------------------------------------------------
# Deep-learning model training (see api/routes_training.py)
# ---------------------------------------------------------------------------
class StartTrainingRequest(BaseModel):
    """Body for `POST /model-training/runs`."""

    symbol: str = Field(
        default="XAUUSD",
        description="Broker symbol to train on. Its M5/M15/H1/H4 candles must already be in "
        "the `candles` table — training reads the database, it does not fetch history.",
        examples=["XAUUSD", "Step Index 200"],
    )


class TrainingRunOut(BaseModel):
    """One training run's lifecycle. Mirrors
    `strategies/application/model_training.TrainingRun`."""

    id: str = Field(description="Run identifier, unique per backend process.")
    symbol: str = Field(description="Symbol this run trained on.")
    model_name: str = Field(
        description="Weights file stem written to `data/ml_models/`, e.g. `smc_dl_m5_v2`."
    )
    state: str = Field(description="One of: running, succeeded, failed.")
    started_at: datetime = Field(description="When the training subprocess was launched (UTC).")
    finished_at: datetime | None = Field(
        default=None, description="When it exited (UTC); null while still running."
    )
    exit_code: int | None = Field(
        default=None, description="Subprocess exit status; null while still running."
    )
    error: str | None = Field(
        default=None, description="Failure reason when `state` is `failed`, else null."
    )
    log_tail: list[str] = Field(
        default_factory=list,
        description="Last lines of the run's combined stdout/stderr, which include the "
        "walk-forward table and the in-sample vs out-of-sample verdict.",
    )

    @classmethod
    def from_domain(cls, run) -> TrainingRunOut:
        return cls(
            id=run.id,
            symbol=run.symbol,
            model_name=run.model_name,
            state=run.state,
            started_at=run.started_at,
            finished_at=run.finished_at,
            exit_code=run.exit_code,
            error=run.error,
            log_tail=list(run.log_tail),
        )


class TrainingSummaryOut(BaseModel):
    """Walk-forward verdict for a trained model — the numbers that decide
    whether it should be switched on."""

    folds: int = Field(default=0, description="Number of chronological folds evaluated.")
    is_avg_r_mean: float | None = Field(
        default=None, description="Mean in-sample average R across folds."
    )
    oos_avg_r_mean: float | None = Field(
        default=None,
        description="Mean out-of-sample average R across folds. This is the number that "
        "matters; a large gap to `is_avg_r_mean` is overfitting.",
    )
    oos_avg_r_min: float | None = Field(
        default=None, description="Worst fold's out-of-sample average R."
    )
    oos_pf_mean: float | None = Field(
        default=None, description="Mean out-of-sample profit factor across folds."
    )
    oos_trades_total: int | None = Field(
        default=None, description="Total trades the gate would have taken out-of-sample."
    )
    folds_positive_oos: int | None = Field(
        default=None, description="How many folds had a positive out-of-sample average R."
    )
    brier_secure_is_mean: float | None = Field(
        default=None, description="In-sample Brier score of the secure head (lower is better)."
    )
    brier_secure_oos_mean: float | None = Field(
        default=None, description="Out-of-sample Brier score of the secure head."
    )
    threshold_tunable: str | None = Field(
        default=None,
        description="Whether the expected-R gate threshold's in-sample and out-of-sample "
        "optima agree. 'NOT TUNABLE' means the parameter must not be fitted on in-sample "
        "results — doing so produced a PF 2.70 in-sample / PF 0.12 out-of-sample filter here.",
    )


class TrainedModelOut(BaseModel):
    """A model on disk. Mirrors
    `strategies/application/model_training.TrainedModel`."""

    model_name: str = Field(description="Weights file stem, e.g. `smc_dl_m5_v2`.")
    symbol: str = Field(description="Symbol the model was trained on.")
    trained_at: datetime | None = Field(
        default=None, description="When training finished (UTC), from the model's metadata."
    )
    n_bars: int = Field(description="Usable training bars after feature/label alignment.")
    data_start: str | None = Field(
        default=None, description="Timestamp of the first bar in the training set."
    )
    data_end: str | None = Field(
        default=None, description="Timestamp of the last bar in the training set."
    )
    feature_count: int = Field(description="Number of input features the weights expect.")
    weights_bytes: int = Field(description="Size of the `.pt` file on disk; 0 if it is missing.")
    summary: TrainingSummaryOut = Field(
        description="Walk-forward verdict — see `TrainingSummaryOut`."
    )

    # `model_` is Pydantic's protected namespace; these fields are model
    # *metadata*, not Pydantic config, so the warning is silenced explicitly.
    model_config = {"protected_namespaces": ()}

    @classmethod
    def from_domain(cls, model) -> TrainedModelOut:
        return cls(
            model_name=model.model_name,
            symbol=model.symbol,
            trained_at=model.trained_at,
            n_bars=model.n_bars,
            data_start=model.data_start,
            data_end=model.data_end,
            feature_count=model.feature_count,
            weights_bytes=model.weights_bytes,
            summary=TrainingSummaryOut(**{
                key: value
                for key, value in (model.summary or {}).items()
                if key in TrainingSummaryOut.model_fields
            }),
        )


class FoldEconomicsOut(BaseModel):
    """Realised R of the trades a fold's gate would have taken. Win rate is
    intentionally absent — see `routes_training.list_models`."""

    trades: int = Field(default=0, description="Non-overlapping trades taken in this fold.")
    sum_r: float = Field(default=0.0, description="Total R booked.")
    avg_r: float = Field(default=0.0, description="Average R per trade.")
    profit_factor: float = Field(default=0.0, description="Gross win R / gross loss R.")
    max_drawdown_r: float = Field(
        default=0.0, description="Worst peak-to-trough equity drop, in R."
    )


class WalkForwardFoldOut(BaseModel):
    """One chronological fold of the walk-forward validation."""

    fold: int = Field(description="1-based fold index, in time order.")
    oos_start: str = Field(description="First out-of-sample bar timestamp.")
    oos_end: str = Field(description="Last out-of-sample bar timestamp.")
    temperature: float = Field(
        description="Calibration temperature fitted on a held-out tail of the training window."
    )
    brier_secure_is: float = Field(description="In-sample Brier score of the secure head.")
    brier_secure_oos: float = Field(description="Out-of-sample Brier score of the secure head.")
    revert_acc_oos: float = Field(
        description="Out-of-sample accuracy of the reversal head against a ~50/50 base rate. "
        "Values near 0.50 mean the head has no skill and exits must not be gated on it."
    )
    base_rate_secure_oos: float = Field(
        description="Fraction of out-of-sample bars that actually secured, for reference "
        "against the 0.833 break-even of the engine's 0.2R trailing rule."
    )
    economics_is: FoldEconomicsOut = Field(description="In-sample trade economics for this fold.")
    economics_oos: FoldEconomicsOut = Field(
        description="Out-of-sample trade economics for this fold — the honest number."
    )

    @classmethod
    def from_meta(cls, fold: dict) -> WalkForwardFoldOut:
        return cls(
            fold=int(fold.get("fold", 0)),
            oos_start=str(fold.get("oos_start", "")),
            oos_end=str(fold.get("oos_end", "")),
            temperature=float(fold.get("temperature", 1.0)),
            brier_secure_is=float(fold.get("brier_secure_is", 0.0)),
            brier_secure_oos=float(fold.get("brier_secure_oos", 0.0)),
            revert_acc_oos=float(fold.get("revert_acc_oos", 0.0)),
            base_rate_secure_oos=float(fold.get("base_rate_secure_oos", 0.0)),
            economics_is=FoldEconomicsOut(**(fold.get("economics_is") or {})),
            economics_oos=FoldEconomicsOut(**(fold.get("economics_oos") or {})),
        )
