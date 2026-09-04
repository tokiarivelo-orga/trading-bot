"""Strategy version persistence (sync SQLAlchemy; call via asyncio.to_thread)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.strategies.adapters.orm import StrategyVersionRow
from src.strategies.domain.versioning import CodeSource, StrategyVersion, VersionStatus


class StrategyVersionRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(self, version: StrategyVersion, account_id: str = "default") -> None:
        row = _to_row(version, account_id)
        with self._session_factory() as session:
            session.merge(row)
            session.commit()

    def get(self, version_id: str) -> StrategyVersion | None:
        with self._session_factory() as session:
            row = session.get(StrategyVersionRow, version_id)
        return _to_domain(row) if row else None

    def delete(self, version_id: str) -> None:
        with self._session_factory() as session:
            row = session.get(StrategyVersionRow, version_id)
            if row is not None:
                session.delete(row)
                session.commit()

    def clear_parent_references(self, version_id: str) -> None:
        """Nulls `parent_version_id` on every row that points at `version_id`,
        so deleting a version never leaves children with a dangling parent
        link (§ StrategyVersionService.delete_version)."""
        with self._session_factory() as session:
            children = session.scalars(
                select(StrategyVersionRow).where(StrategyVersionRow.parent_version_id == version_id)
            ).all()
            for child in children:
                child.parent_version_id = None
            session.commit()

    def list_all(
        self,
        name: str | None = None,
        status: VersionStatus | None = None,
        account_id: str = "default",
    ) -> list[StrategyVersion]:
        query = (
            select(StrategyVersionRow)
            .where(StrategyVersionRow.account_id == account_id)
            .order_by(StrategyVersionRow.name, StrategyVersionRow.version.desc())
        )
        if name is not None:
            query = query.where(StrategyVersionRow.name == name)
        if status is not None:
            query = query.where(StrategyVersionRow.status == status.value)
        with self._session_factory() as session:
            rows = session.scalars(query).all()
        return [_to_domain(row) for row in rows]

    def latest_version_number(self, name: str, account_id: str = "default") -> int:
        with self._session_factory() as session:
            rows = session.scalars(
                select(StrategyVersionRow.version).where(
                    StrategyVersionRow.name == name, StrategyVersionRow.account_id == account_id
                )
            ).all()
        return max(rows, default=0)

    def get_active(self, name: str, account_id: str = "default") -> StrategyVersion | None:
        query = select(StrategyVersionRow).where(
            StrategyVersionRow.name == name,
            StrategyVersionRow.status == VersionStatus.ACTIVE,
            StrategyVersionRow.account_id == account_id,
        )
        with self._session_factory() as session:
            row = session.scalars(query).first()
        return _to_domain(row) if row else None

    def list_active(self, account_id: str = "default") -> list[StrategyVersion]:
        query = select(StrategyVersionRow).where(
            StrategyVersionRow.status == VersionStatus.ACTIVE,
            StrategyVersionRow.account_id == account_id,
        )
        with self._session_factory() as session:
            rows = session.scalars(query).all()
        return [_to_domain(row) for row in rows]


def _to_row(version: StrategyVersion, account_id: str) -> StrategyVersionRow:
    return StrategyVersionRow(
        id=version.id,
        account_id=account_id,
        name=version.name,
        version=version.version,
        file_path=version.file_path,
        code_hash=version.code_hash,
        source=version.source.value,
        status=version.status.value,
        created_at=int(version.created_at.timestamp()),
        parent_version_id=version.parent_version_id,
        draft_id=version.draft_id,
        spec=version.spec,
        backtest_report_id=version.backtest_report_id,
        paused=version.paused,
    )


def _to_domain(row: StrategyVersionRow) -> StrategyVersion:
    return StrategyVersion(
        id=row.id,
        name=row.name,
        version=row.version,
        file_path=row.file_path,
        code_hash=row.code_hash,
        source=CodeSource(row.source),
        status=VersionStatus(row.status),
        created_at=datetime.fromtimestamp(row.created_at, tz=UTC),
        parent_version_id=row.parent_version_id,
        draft_id=row.draft_id,
        spec=row.spec,
        backtest_report_id=row.backtest_report_id,
        paused=row.paused,
    )
