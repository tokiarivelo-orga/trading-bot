"""Trigger and observe deep-learning model training runs.

Training is a long CPU job (minutes, and it holds the whole M5 history in
memory), so it runs as a **subprocess** rather than inside the API worker.
Three properties fall out of that choice and all three are wanted:

* a training crash cannot take the backend down with it;
* the GIL-bound training loop cannot stall the event loop that is also
  running the trade engine;
* the exact same entry point the cron script uses
  (``scripts/train_smc_dl_v2.py``) is what the button runs, so "it worked
  from the shell" and "it worked from the UI" cannot diverge.

It also keeps ``src/`` from importing ``scripts/``, which would invert the
dependency direction the rest of this codebase maintains.

HISTORY COMES FROM THE ARTIFACTS, NOT A TABLE
────────────────────────────────────────────────────────────────────────
Every finished run writes ``<model>_meta.json`` next to the weights,
carrying the trained-at timestamp, the walk-forward folds, calibration and
the honest IS/OOS summary. That file *is* the history, so this service reads
it rather than duplicating the same facts into a new table that could then
disagree with the weights actually on disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("data") / "ml_models"
DEFAULT_SCRIPT = Path("scripts") / "train_smc_dl_v2.py"
# Enough to show the walk-forward table and the verdict in the UI without
# streaming an unbounded log into memory.
_LOG_TAIL_LINES = 400


class TrainingAlreadyRunningError(RuntimeError):
    """A second run was requested while one is still in flight.

    Rejected rather than queued: two concurrent runs would race on the same
    ``.pt``/``_scaler.npz``/``_meta.json`` triple and could leave weights
    that do not match their own scaler.
    """


@dataclass
class TrainingRun:
    id: str
    symbol: str
    model_name: str
    state: str  # running | succeeded | failed
    started_at: datetime
    finished_at: datetime | None = None
    exit_code: int | None = None
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainedModel:
    model_name: str
    symbol: str
    trained_at: datetime | None
    n_bars: int
    data_start: str | None
    data_end: str | None
    feature_count: int
    summary: dict
    weights_bytes: int


def model_name_for(symbol: str) -> str:
    """Weights file stem for a symbol. Mirrors ``train_smc_dl_v2.main``."""
    if symbol == "XAUUSD":
        return "smc_dl_m5_v2"
    return "smc_dl_m5_v2_" + symbol.replace(" ", "_").lower()


class ModelTrainingService:
    def __init__(
        self,
        *,
        repo_root: Path,
        models_dir: Path | None = None,
        script: Path | None = None,
        folds: int = 4,
        epochs: int = 8,
    ) -> None:
        self._repo_root = repo_root
        self._models_dir = models_dir or (repo_root / DEFAULT_MODELS_DIR)
        self._script = script or (repo_root / DEFAULT_SCRIPT)
        self._folds = folds
        self._epochs = epochs
        self._current: TrainingRun | None = None
        self._recent: list[TrainingRun] = []
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    # -- commands ------------------------------------------------------
    async def start(self, symbol: str) -> TrainingRun:
        async with self._lock:
            if self._current is not None and self._current.state == "running":
                raise TrainingAlreadyRunningError(
                    f"a training run for {self._current.symbol} is already in progress"
                )
            run = TrainingRun(
                id=str(uuid.uuid4()),
                symbol=symbol,
                model_name=model_name_for(symbol),
                state="running",
                started_at=datetime.now(UTC),
            )
            self._current = run
            self._task = asyncio.create_task(self._run(run), name=f"train-{run.model_name}")
            logger.info("model training started: run=%s symbol=%s", run.id, symbol)
            return run

    async def _run(self, run: TrainingRun) -> None:
        command = [
            sys.executable,
            str(self._script),
            "--symbol",
            run.symbol,
            "--folds",
            str(self._folds),
            "--epochs",
            str(self._epochs),
        ]
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=str(self._repo_root),
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "PYTHONPATH": str(self._repo_root)},
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            run.log_tail = output.splitlines()[-_LOG_TAIL_LINES:]
            run.exit_code = completed.returncode
            run.state = "succeeded" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                run.error = f"training exited with code {completed.returncode}"
                logger.error("model training failed: run=%s rc=%s", run.id, completed.returncode)
            else:
                logger.info("model training finished: run=%s symbol=%s", run.id, run.symbol)
        except Exception as exc:  # noqa: BLE001 — surfaced through the API, not swallowed
            run.state = "failed"
            run.error = repr(exc)
            logger.exception("model training raised: run=%s", run.id)
        finally:
            run.finished_at = datetime.now(UTC)
            self._recent.insert(0, run)
            del self._recent[10:]

    async def stop(self) -> None:
        """Cancel a run in flight — used on shutdown. The subprocess itself is
        left to finish; killing it midway could leave a half-written weights
        file that the strategy would then load."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    # -- queries -------------------------------------------------------
    def current(self) -> TrainingRun | None:
        return self._current

    def recent_runs(self) -> list[TrainingRun]:
        return list(self._recent)

    def trained_models(self) -> list[TrainedModel]:
        """Every model on disk, newest first, read from its ``_meta.json``."""
        if not self._models_dir.exists():
            return []
        models: list[TrainedModel] = []
        for meta_path in sorted(self._models_dir.glob("*_meta.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                logger.warning("unreadable model metadata: %s", meta_path)
                continue
            weights = self._models_dir / (meta.get("model_name", "") + ".pt")
            models.append(
                TrainedModel(
                    model_name=meta.get("model_name", meta_path.stem),
                    symbol=meta.get("symbol", "unknown"),
                    trained_at=_parse_time(meta.get("trained_at")),
                    n_bars=int(meta.get("n_bars", 0)),
                    data_start=meta.get("data_start"),
                    data_end=meta.get("data_end"),
                    feature_count=int(meta.get("input_dim", 0)),
                    summary=meta.get("summary", {}),
                    weights_bytes=weights.stat().st_size if weights.exists() else 0,
                )
            )
        models.sort(key=lambda m: m.trained_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return models

    def walk_forward(self, model_name: str) -> list[dict]:
        """The per-fold IS/OOS detail behind a model — the numbers that decide
        whether it should be trusted, not just that it exists."""
        meta_path = self._models_dir / (model_name + "_meta.json")
        if not meta_path.exists():
            return []
        try:
            return json.loads(meta_path.read_text()).get("walk_forward", [])
        except (OSError, json.JSONDecodeError):
            return []


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
