"""Unit tests for installer/env_writer.py.

Runs only against temp-directory fixtures (copies of the repo's real
`.env.example`) -- never against this repo's actual `.env`, which holds this
developer's real secrets. See CLAUDE.md / the installer task brief.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import env_writer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"


class _FakeAccount:
    """Minimal duck-typed stand-in for wizard.Account -- env_writer.plan()
    reads `.gateway_secret_env` (already decided by the wizard) off each
    account, defaulting to the primary var like wizard.Account's true-primary
    case."""

    def __init__(self, account_id: str, gateway_secret_env: str | None = None):
        self.id = account_id
        self.gateway_secret_env = gateway_secret_env or env_writer.PRIMARY_SECRET_VAR


class EnvWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.env_example_copy = self.tmp_path / ".env.example"
        shutil.copy(ENV_EXAMPLE, self.env_example_copy)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_creates_env_from_example_with_fresh_primary_secret(self) -> None:
        install_dir = self.tmp_path / "install"
        install_dir.mkdir()

        changes = env_writer.plan(install_dir, self.env_example_copy, accounts=[], wine_prefix=None)
        env_writer.apply(install_dir, self.env_example_copy, changes)

        env_path = install_dir / ".env"
        self.assertTrue(env_path.exists())
        values = env_writer.read_existing(env_path)

        self.assertIn("TB_GATEWAY_SHARED_SECRET", values)
        self.assertNotEqual(values["TB_GATEWAY_SHARED_SECRET"], "change-me-long-random-string")
        self.assertEqual(len(values["TB_GATEWAY_SHARED_SECRET"]), 64)  # secrets.token_hex(32)
        # other .env.example content carried over untouched
        self.assertEqual(values["TB_DATABASE_URL"], "sqlite+aiosqlite:///./data/trading.db")

    def test_appends_new_account_secret_without_touching_existing_vars(self) -> None:
        install_dir = self.tmp_path / "install2"
        install_dir.mkdir()
        env_path = install_dir / ".env"
        env_path.write_text(
            "TB_DATABASE_URL=sqlite+aiosqlite:///./data/trading.db\n"
            "TB_GATEWAY_URL=http://127.0.0.1:8787\n"
            "TB_GATEWAY_SHARED_SECRET=already-set-do-not-touch\n"
        )

        accounts = [
            _FakeAccount("default"),  # true primary -- reuses TB_GATEWAY_SHARED_SECRET, already present
            _FakeAccount("ftmo-1", env_writer.account_secret_var("ftmo-1")),
        ]
        changes = env_writer.plan(install_dir, self.env_example_copy, accounts=accounts, wine_prefix=None)
        env_writer.apply(install_dir, self.env_example_copy, changes)

        values = env_writer.read_existing(env_path)
        # pre-existing vars: byte-for-byte untouched
        self.assertEqual(values["TB_GATEWAY_SHARED_SECRET"], "already-set-do-not-touch")
        self.assertEqual(values["TB_GATEWAY_URL"], "http://127.0.0.1:8787")
        self.assertEqual(values["TB_DATABASE_URL"], "sqlite+aiosqlite:///./data/trading.db")
        # new per-account secret appended for the account beyond the first
        self.assertIn("TB_GATEWAY_SHARED_SECRET_FTMO_1", values)
        self.assertEqual(len(values["TB_GATEWAY_SHARED_SECRET_FTMO_1"]), 64)

    def test_never_overwrites_an_existing_account_secret(self) -> None:
        install_dir = self.tmp_path / "install3"
        install_dir.mkdir()
        env_path = install_dir / ".env"
        env_path.write_text(
            "TB_GATEWAY_SHARED_SECRET=primary\nTB_GATEWAY_SHARED_SECRET_FTMO_1=keep-me\n"
        )

        accounts = [
            _FakeAccount("default"),
            _FakeAccount("ftmo-1", env_writer.account_secret_var("ftmo-1")),
        ]
        changes = env_writer.plan(install_dir, self.env_example_copy, accounts=accounts, wine_prefix=None)
        # no change should even be queued for an already-present var
        self.assertFalse(any(c.var_name == "TB_GATEWAY_SHARED_SECRET_FTMO_1" for c in changes))

        env_writer.apply(install_dir, self.env_example_copy, changes)
        values = env_writer.read_existing(env_path)
        self.assertEqual(values["TB_GATEWAY_SHARED_SECRET_FTMO_1"], "keep-me")
        self.assertEqual(values["TB_GATEWAY_SHARED_SECRET"], "primary")

    def test_wine_prefix_written_when_provided(self) -> None:
        install_dir = self.tmp_path / "install4"
        install_dir.mkdir()

        changes = env_writer.plan(install_dir, self.env_example_copy, accounts=[], wine_prefix="/opt/mt5-wine")
        env_writer.apply(install_dir, self.env_example_copy, changes)

        values = env_writer.read_existing(install_dir / ".env")
        self.assertEqual(values["TB_WINEPREFIX"], "/opt/mt5-wine")

    def test_wine_prefix_uncomments_the_template_line(self) -> None:
        install_dir = self.tmp_path / "install5"
        install_dir.mkdir()
        env_path = install_dir / ".env"
        env_path.write_text("TB_GATEWAY_SHARED_SECRET=x\n# TB_WINEPREFIX=/home/user/.mt5\n")

        changes = env_writer.plan(install_dir, self.env_example_copy, accounts=[], wine_prefix="/srv/mt5")
        env_writer.apply(install_dir, self.env_example_copy, changes)

        text = env_path.read_text()
        self.assertIn("TB_WINEPREFIX=/srv/mt5\n", text)
        self.assertNotIn("# TB_WINEPREFIX=/home/user/.mt5", text)
        # exactly one TB_WINEPREFIX line -- the commented template line was replaced, not duplicated
        self.assertEqual(text.count("TB_WINEPREFIX="), 1)

    def test_first_in_session_account_gets_its_own_secret_when_primary_already_exists(self) -> None:
        # Regression test: a custom-id account that happens to be first in
        # *this wizard session* must still get its own suffixed secret var
        # (not silently reuse TB_GATEWAY_SHARED_SECRET) whenever the file
        # already had an existing primary account before this run -- wizard.py
        # decides this via `gateway_secret_env`, and plan() must honor that
        # per-account decision rather than assuming "first in the list ==
        # primary account".
        install_dir = self.tmp_path / "install6"
        install_dir.mkdir()
        env_path = install_dir / ".env"
        env_path.write_text("TB_GATEWAY_SHARED_SECRET=primary-already-set\n")

        # Only one account in this session's list, at position 0, but it is
        # NOT the true primary (gateway_secret_env is already the suffixed
        # var, exactly as wizard.py would compute when existing_account_ids
        # is non-empty).
        accounts = [_FakeAccount("ftmo-1", env_writer.account_secret_var("ftmo-1"))]
        changes = env_writer.plan(install_dir, self.env_example_copy, accounts=accounts, wine_prefix=None)
        env_writer.apply(install_dir, self.env_example_copy, changes)

        values = env_writer.read_existing(env_path)
        self.assertEqual(values["TB_GATEWAY_SHARED_SECRET"], "primary-already-set")  # untouched
        self.assertIn("TB_GATEWAY_SHARED_SECRET_FTMO_1", values)  # its own secret was generated
        self.assertEqual(len(values["TB_GATEWAY_SHARED_SECRET_FTMO_1"]), 64)

    def test_account_secret_var_slug(self) -> None:
        self.assertEqual(env_writer.account_secret_var("ftmo-1"), "TB_GATEWAY_SHARED_SECRET_FTMO_1")
        self.assertEqual(env_writer.account_secret_var("Prop Firm A"), "TB_GATEWAY_SHARED_SECRET_PROP_FIRM_A")


if __name__ == "__main__":
    unittest.main()
