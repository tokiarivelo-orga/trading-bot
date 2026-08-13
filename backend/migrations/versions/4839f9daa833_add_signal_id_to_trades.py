"""add signal_id to trades

Nullable/additive: joins a journaled trade back to the `SignalDecision`
(`signal_decisions` table) that led to it, and transitively to the
`order_book_snapshots` row captured for the same signal (order_book/ Phase 5).
Null for every trade journaled before this migration, and for trades opened
manually or via the API, which have no signal behind them.

Revision ID: 4839f9daa833
Revises: 5e28fd5356ff
Create Date: 2026-08-13 15:41:23.181537

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '4839f9daa833'
down_revision = '5e28fd5356ff'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('trades', sa.Column('signal_id', sa.String(length=36), nullable=True))
    op.create_index('ix_trades_signal_id', 'trades', ['signal_id'])


def downgrade() -> None:
    op.drop_index('ix_trades_signal_id', table_name='trades')
    op.drop_column('trades', 'signal_id')
