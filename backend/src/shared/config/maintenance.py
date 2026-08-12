"""Housekeeping config for tables/files that grow unbounded on SQLite —
not business logic, hence living alongside `Settings` in shared/config
rather than in a domain module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class MaintenanceConfig:
    activity_log_retention_days: int
    activity_log_check_interval_hours: float
    wal_checkpoint_enabled: bool
    wal_checkpoint_interval_minutes: float
    # Bi-weekly deep-learning retraining — see
    # `strategies/application/training_scheduler.py` for why the schedule is
    # an interval rather than a cron expression.
    model_training_enabled: bool
    model_training_interval_days: float
    model_training_on_startup: bool
    model_training_symbols: tuple[str, ...]
