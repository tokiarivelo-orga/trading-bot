from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import module ORM models here as they land, so autogenerate sees them.
from src.activity.adapters import orm as activity_orm  # noqa: F401
from src.ai.adapters import orm as ai_orm  # noqa: F401
from src.indicators.adapters import orm as indicators_orm  # noqa: F401
from src.journal.adapters import orm as journal_orm  # noqa: F401
from src.market_data.adapters import orm as market_data_orm  # noqa: F401
from src.news.adapters import orm as news_orm  # noqa: F401
from src.shared.config.settings import Settings
from src.shared.db.base import Base
from src.strategies.adapters import orm as strategies_orm  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = Settings()
config.set_main_option("sqlalchemy.url", settings.database_url.replace("+aiosqlite", ""))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
