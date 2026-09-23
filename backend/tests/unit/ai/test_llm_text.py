"""`llm_text.py` recovers Python/JSON payloads from LLM completions that
don't obey a prompt's "no fences, no commentary" instruction — the two
regressions below are lifted from real sandbox rejections
("unterminated string literal (detected at line 1)" and "invalid syntax
(<unknown>, line 2)") caused by a stray preamble line ahead of the code."""

from __future__ import annotations

from src.ai.application.llm_text import extract_python_code, strip_fences

_CODE = (
    "from src.strategies.domain.models import MarketContext, StrategySpec\n\n\n"
    "class BreakoutStrategy:\n"
    "    def __init__(self):\n"
    '        self.spec = StrategySpec(name="breakout", version=1, symbols=("XAUUSD",), '
    'entry_timeframe="M5", confirmation_timeframes=(), params={})\n\n'
    "    def evaluate(self, ctx: MarketContext):\n"
    "        return None\n"
)


def test_extract_python_code_plain():
    assert extract_python_code(_CODE) == _CODE.strip()


def test_extract_python_code_fenced():
    raw = f"```python\n{_CODE}```"
    assert extract_python_code(raw) == _CODE.strip()


def test_extract_python_code_strips_apostrophe_title_preamble():
    # A one-line title wrapped in single quotes ahead of the code: the
    # apostrophe in "Trader's" closes the quote early, leaving a second,
    # genuinely unterminated string on line 1 — this is what produced
    # "unterminated string literal (detected at line 1)" from the sandbox.
    raw = f"'Trader's Breakout Strategy'\n\n{_CODE}"
    assert extract_python_code(raw) == _CODE.strip()


def test_extract_python_code_strips_sentence_preamble():
    # A one-line summary sentence ahead of the code, no fence: this is what
    # produced "invalid syntax (<unknown>, line 2)" — line 1 (the sentence)
    # doesn't parse as a statement, so `ast.parse` reports the failure a
    # line further in.
    raw = f"Here is the strategy you asked for:\n\n{_CODE}"
    assert extract_python_code(raw) == _CODE.strip()


def test_extract_python_code_fenced_with_surrounding_commentary():
    raw = f"Sure, here's the code:\n\n```python\n{_CODE}```\n\nLet me know if you need changes."
    assert extract_python_code(raw) == _CODE.strip()


def test_strip_fences_no_fence_returns_trimmed_text():
    assert strip_fences('  {"a": 1}  \n') == '{"a": 1}'


def test_strip_fences_json_fence():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
