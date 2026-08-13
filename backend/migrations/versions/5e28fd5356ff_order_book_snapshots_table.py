"""order book snapshots table

Per-signal market-depth (order-book) snapshots for AI-training data export
(order_book/ Phase 4 — a vertical slice, not yet wired into the trade loop).
One row per signal that had real depth to capture; symbols/brokers that
report no depth (the common case for OTC CFD/synthetic instruments like
XAUUSD, VIX75, Boom) simply have no row — absence is the graceful-
degradation signal, not a nullable "empty" row.

Revision ID: 5e28fd5356ff
Revises: c82842a85999
Create Date: 2026-08-13 15:19:41.599155

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = '5e28fd5356ff'
down_revision = 'c82842a85999'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'order_book_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('account_id', sa.String(length=64), nullable=False),
        sa.Column('signal_id', sa.String(length=64), nullable=False),
        sa.Column('symbol', sa.String(length=16), nullable=False),
        sa.Column('captured_at', sa.Integer(), nullable=False),
        sa.Column('levels', sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_order_book_snapshots_account_id'),
        'order_book_snapshots',
        ['account_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_order_book_snapshots_signal_id'),
        'order_book_snapshots',
        ['signal_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_order_book_snapshots_symbol'),
        'order_book_snapshots',
        ['symbol'],
        unique=False,
    )
    op.create_index(
        'ix_order_book_snapshots_account_signal',
        'order_book_snapshots',
        ['account_id', 'signal_id'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_order_book_snapshots_account_signal', table_name='order_book_snapshots')
    op.drop_index(
        op.f('ix_order_book_snapshots_symbol'), table_name='order_book_snapshots'
    )
    op.drop_index(
        op.f('ix_order_book_snapshots_signal_id'), table_name='order_book_snapshots'
    )
    op.drop_index(
        op.f('ix_order_book_snapshots_account_id'), table_name='order_book_snapshots'
    )
    op.drop_table('order_book_snapshots')
