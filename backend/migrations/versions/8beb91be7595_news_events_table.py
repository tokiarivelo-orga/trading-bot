"""news events table

Persisted economic-calendar events (Phase 6 Part B). `NewsWindowService`
only ever kept fetched events in an in-memory list (`_events`) for its
zero-I/O hot-path lookup (`active_window_for`, called every M5 close) — a
restart loses all calendar history, and no other source of truth in this
codebase can reconstruct it (the calendar adapters, Finnhub/ForexFactory,
only ever return the *current* upcoming window, not history). This table
exists specifically to stop losing it: every calendar refresh now also
upserts its batch here, keyed on `(name, time)` so re-fetching the same
upcoming events never duplicates rows.

`balance_before`/`balance_after` (Phase 6 Part C) are nullable enrichment
columns: the account balance immediately before/after this event's news
window opened/closed, stamped as `NewsWindowEntered`/`NewsWindowExited`
fire. Both start and can stay null — an event that never activates a
tracked news window, or a window that never cleanly exits (process restart
mid-window), simply leaves them unstamped rather than blocking anything.

Revision ID: 8beb91be7595
Revises: 4839f9daa833
Create Date: 2026-08-13 20:01:43.770985

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '8beb91be7595'
down_revision = '4839f9daa833'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'news_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('time', sa.Integer(), nullable=False),
        sa.Column('impact', sa.String(length=16), nullable=False),
        sa.Column('currency', sa.String(length=16), nullable=False),
        sa.Column('forecast', sa.String(length=64), nullable=True),
        sa.Column('previous', sa.String(length=64), nullable=True),
        sa.Column('actual', sa.String(length=64), nullable=True),
        sa.Column('balance_before', sa.Float(), nullable=True),
        sa.Column('balance_after', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_news_events_time'), 'news_events', ['time'], unique=False
    )
    op.create_index(
        op.f('ix_news_events_impact'), 'news_events', ['impact'], unique=False
    )
    op.create_index(
        'ix_news_events_name_time', 'news_events', ['name', 'time'], unique=True
    )


def downgrade() -> None:
    op.drop_index('ix_news_events_name_time', table_name='news_events')
    op.drop_index(op.f('ix_news_events_impact'), table_name='news_events')
    op.drop_index(op.f('ix_news_events_time'), table_name='news_events')
    op.drop_table('news_events')
