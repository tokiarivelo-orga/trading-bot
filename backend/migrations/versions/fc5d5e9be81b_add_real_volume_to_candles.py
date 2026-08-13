"""add real_volume to candles

MT5's rate rows carry a `real_volume` field (actual traded volume, distinct
from `tick_volume`'s tick count) that the gateway/backend previously read
and discarded. This adds a nullable, go-forward-only column: historical
rows were written before this field was captured off the wire and never
had a value for it, so NULL has to stay distinguishable from a genuine 0
reported by a broker that doesn't populate real_volume for a symbol. No
backfill of existing rows.

Revision ID: fc5d5e9be81b
Revises: b4c5d6e7f8a9
Create Date: 2026-08-13 14:49:55.719167

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = 'fc5d5e9be81b'
down_revision = 'b4c5d6e7f8a9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('candles', sa.Column('real_volume', sa.BigInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column('candles', 'real_volume')
