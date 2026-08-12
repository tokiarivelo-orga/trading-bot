"""Bi-weekly in-app retraining.

``scripts/cron_train_models.sh`` already exists and still works, but a cron
entry is invisible to the app: nothing in the UI can say when the last run
happened or whether it failed, and it silently does nothing on a machine
where crontab was never installed. This runs the same subprocess from inside
the backend, through the same ``ModelTrainingService`` the API button uses,
so a scheduled run and a manual run are the same object with the same
history.

Follows the pattern of the other periodic services here
(``shared/db/checkpoint.py``, ``activity/application/retention_service.py``):
a single asyncio task, started from the lifespan, cancelled on shutdown.

The interval is a plain sleep rather than a wall-clock cron expression on
purpose — the backend restarts often enough during development that
"fourteen days since this process last trained" is both simpler and closer to
what is wanted than "the 1st and 15th at 03:00", and it cannot pile up
missed runs after downtime.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from src.strategies.application.model_training import (
    ModelTrainingService,
    TrainingAlreadyRunningError,
)

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_DAYS = 14.0


class TrainingScheduler:
    def __init__(
        self,
        service: ModelTrainingService,
        symbols: list[str],
        *,
        interval_days: float = DEFAULT_INTERVAL_DAYS,
        enabled: bool = True,
        train_on_startup: bool = False,
    ) -> None:
        self._service = service
        self._symbols = symbols
        self._interval_s = max(interval_days, 0.01) * 86400.0
        self._enabled = enabled
        self._train_on_startup = train_on_startup
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if not self._enabled:
            logger.info("model retraining disabled (configs/maintenance.yaml model_training)")
            return
        if not self._symbols:
            logger.info("model retraining skipped: no symbols configured")
            return
        self._task = asyncio.create_task(self._run(), name="model-retraining")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        # A backend restart must not kick off a multi-minute CPU job while the
        # trade engine is still coming up.
        if not self._train_on_startup:
            await asyncio.sleep(self._interval_s)
        while True:
            for symbol in self._symbols:
                await self._train_once(symbol)
            await asyncio.sleep(self._interval_s)

    async def _train_once(self, symbol: str) -> None:
        try:
            run = await self._service.start(symbol)
        except TrainingAlreadyRunningError:
            logger.info("scheduled retraining skipped for %s: a run is already active", symbol)
            return
        except Exception:
            logger.exception("scheduled retraining could not be started for %s", symbol)
            return

        logger.info("scheduled retraining started: symbol=%s run=%s", symbol, run.id)
        # Serialise the symbols: two training subprocesses at once would
        # contend for the same CPU the trade engine needs, on top of the
        # weights race `ModelTrainingService.start` already rejects.
        while run.state == "running":
            await asyncio.sleep(5.0)
        logger.info(
            "scheduled retraining finished: symbol=%s run=%s state=%s", symbol, run.id, run.state
        )
