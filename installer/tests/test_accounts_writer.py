"""Unit tests for installer/accounts_writer.py.

Runs only against temp-directory fixtures (a copy of the repo's real
configs/accounts.yaml) -- never against this repo's actual
configs/accounts.yaml. See CLAUDE.md / the installer task brief.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import accounts_writer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACCOUNTS_FIXTURE = REPO_ROOT / "configs" / "accounts.yaml"


class _FakeAccount:
    """Minimal duck-typed stand-in for wizard.Account."""

    def __init__(
        self,
        account_id: str,
        label: str = "Test account",
        mode: str = "paper",
        gateway_host: str = "127.0.0.1",
        gateway_port: int = 8790,
        terminal_path: str | None = None,
        gateway_secret_env: str = "TB_GATEWAY_SHARED_SECRET_TEST",
    ):
        self.id = account_id
        self.label = label
        self.mode = mode
        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
        self.terminal_path = terminal_path
        self.gateway_secret_env = gateway_secret_env

    @property
    def gateway_url(self) -> str:
        return f"http://{self.gateway_host}:{self.gateway_port}"


class AccountsWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.accounts_path = self.tmp_path / "accounts.yaml"
        shutil.copy(ACCOUNTS_FIXTURE, self.accounts_path)
        self.original_text = self.accounts_path.read_text()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_existing_ids_finds_the_fixture_accounts(self) -> None:
        self.assertEqual(set(accounts_writer.existing_ids(self.accounts_path)), {"default", "demo-1"})

    def test_existing_gateway_ports_finds_the_fixture_ports(self) -> None:
        self.assertEqual(set(accounts_writer.existing_gateway_ports(self.accounts_path)), {8787, 8788})

    def test_regex_fallback_agrees_with_pyyaml_result(self) -> None:
        # sanity: both id-detection paths must agree on the fixture, since
        # accounts_writer falls back to the regex scan whenever PyYAML isn't
        # importable in the environment running the real installer.
        self.assertEqual(set(accounts_writer._regex_existing_ids(self.accounts_path)), {"default", "demo-1"})

    def test_appends_new_account_with_correct_indentation(self) -> None:
        account = _FakeAccount(
            "ftmo-1",
            label="FTMO live",
            mode="live",
            gateway_port=8790,
            terminal_path="/home/user/.mt5-ftmo/drive_c/Program Files/MetaTrader 5/terminal64.exe",
        )
        changes = accounts_writer.plan(self.accounts_path, [account])
        self.assertEqual(len(changes), 1)
        self.assertFalse(changes[0].skipped)

        accounts_writer.apply(self.accounts_path, changes)

        text = self.accounts_path.read_text()
        self.assertTrue(text.startswith(self.original_text))
        appended = text[len(self.original_text) :]
        lines = [line for line in appended.splitlines() if line.strip()]

        self.assertEqual(lines[0], "  - id: ftmo-1")
        self.assertEqual(lines[1], '    label: "FTMO live"')
        self.assertEqual(lines[2], "    gateway_url: http://127.0.0.1:8790")
        self.assertEqual(lines[3], "    gateway_shared_secret_env: TB_GATEWAY_SHARED_SECRET_TEST")
        self.assertEqual(lines[4], "    mode: live # paper | live")
        self.assertEqual(lines[5], "    enabled: true")
        self.assertEqual(lines[6], "    risk_override_file: null")
        self.assertEqual(
            lines[7],
            '    mt5_terminal_path: "/home/user/.mt5-ftmo/drive_c/Program Files/MetaTrader 5/terminal64.exe"',
        )
        # existing entries (default, demo-1) must be completely unmodified
        self.assertIn('  - id: default', text)
        self.assertIn('  - id: demo-1', text)

    def test_omits_terminal_path_line_when_not_provided(self) -> None:
        account = _FakeAccount("account-2")
        changes = accounts_writer.plan(self.accounts_path, [account])
        accounts_writer.apply(self.accounts_path, changes)

        appended = self.accounts_path.read_text()[len(self.original_text) :]
        self.assertNotIn("mt5_terminal_path", appended)
        self.assertNotIn("mt5_terminal_subpath", appended)

    def test_refuses_to_duplicate_an_existing_id(self) -> None:
        account = _FakeAccount("default")  # already present in the fixture

        changes = accounts_writer.plan(self.accounts_path, [account])
        self.assertEqual(len(changes), 1)
        self.assertTrue(changes[0].skipped)
        self.assertIn("already exists in accounts.yaml, leaving it untouched", changes[0].skip_reason)

        accounts_writer.apply(self.accounts_path, changes)
        # file byte-for-byte unchanged
        self.assertEqual(self.accounts_path.read_text(), self.original_text)

    def test_refuses_to_duplicate_an_id_added_earlier_in_the_same_run(self) -> None:
        accounts = [_FakeAccount("new-1"), _FakeAccount("new-1")]
        changes = accounts_writer.plan(self.accounts_path, accounts)
        self.assertFalse(changes[0].skipped)
        self.assertTrue(changes[1].skipped)

        accounts_writer.apply(self.accounts_path, changes)
        text = self.accounts_path.read_text()
        self.assertEqual(text.count("- id: new-1"), 1)

    def test_mixed_batch_appends_new_and_skips_existing(self) -> None:
        accounts = [_FakeAccount("demo-1"), _FakeAccount("new-account")]
        changes = accounts_writer.plan(self.accounts_path, accounts)
        self.assertTrue(changes[0].skipped)
        self.assertFalse(changes[1].skipped)

        accounts_writer.apply(self.accounts_path, changes)
        text = self.accounts_path.read_text()
        self.assertEqual(text.count("- id: demo-1"), 1)  # not duplicated
        self.assertEqual(text.count("- id: new-account"), 1)  # appended


if __name__ == "__main__":
    unittest.main()
