"""add atr and day_of_week to candles

Two nullable, go-forward columns for per-candle enrichment: `atr_14` (14-
period ATR computed from the bar and its trailing history) and `day_of_week`
(0=Monday..6=Sunday, UTC). Both are nullable because this migration only
adds the columns — it does not populate them. Computing ATR needs real
trailing history that isn't available mid-transaction for every bar in a
large table, and blocking a schema change on that work would turn a normally
fast/lock-free `ADD COLUMN` into a long-running write lock. Instead,
`scripts/backfill_candle_enrichment.py` fills both columns in afterward
(idempotent, safely re-runnable), the same way `real_volume` was added
nullable and left for callers to populate.

Revision ID: c82842a85999
Revises: fc5d5e9be81b
Create Date: 2026-08-13 15:03:39.904287

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = 'c82842a85999'
down_revision = 'fc5d5e9be81b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('candles', sa.Column('atr_14', sa.Float(), nullable=True))
    op.add_column('candles', sa.Column('day_of_week', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('candles', 'day_of_week')
    op.drop_column('candles', 'atr_14')
