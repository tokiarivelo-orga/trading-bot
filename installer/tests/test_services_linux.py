"""Unit tests for installer/services_linux.py.

Only ever exercises pure rendering functions and dry_run=True / tmp-dir
writes with resolve_tool_paths() and subprocess.run patched out -- this
machine's real systemd --user session (~/.config/systemd/user/) and its
real uv/pnpm/wine installs must never be touched by these tests. See
installer/tests/test_env_writer.py / test_accounts_writer.py for the
existing style this follows.
"""

from __future__ import annotations

import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import services_linux  # noqa: E402

FAKE_TOOLS = services_linux.ToolPaths(
    uv_bin="/fake/bin/uv",
    pnpm_bin="/fake/bin/pnpm",
    wine_bin="/fake/bin/wine",
    wine_python="/fake/.mt5/drive_c/users/dev/AppData/Local/Programs/Python/Python312/python.exe",
)


def _fake_state(**overrides) -> dict:
    state = {
        "install_dir": "/opt/trading-bot",
        "wine_prefix": "/home/dev/.mt5",
        "run_wine_setup": False,
        "autostart_on_boot": True,
        "accounts": [
            {
                "id": "default",
                "label": "Primary MT5 account",
                "mode": "paper",
                "gateway_host": "127.0.0.1",
                "gateway_port": 8787,
                "terminal_path": "/home/dev/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe",
                "gateway_secret_env": "TB_GATEWAY_SHARED_SECRET",
            },
            {
                "id": "ftmo-1",
                "label": "FTMO account",
                "mode": "live",
                "gateway_host": "127.0.0.1",
                "gateway_port": 8788,
                "terminal_path": None,
                "gateway_secret_env": "TB_GATEWAY_SHARED_SECRET_FTMO_1",
            },
        ],
    }
    state.update(overrides)
    return state


FAKE_ENV_VALUES = {
    "TB_GATEWAY_SHARED_SECRET": "primary-secret-value",
    "TB_GATEWAY_SHARED_SECRET_FTMO_1": "ftmo-secret-value",
    "TB_FRONTEND_PORT": "3001",
}


class ResolveToolPathsTests(unittest.TestCase):
    def test_resolves_every_tool_to_an_absolute_path(self) -> None:
        fake_which = {
            "uv": "/usr/local/bin/uv",
            "pnpm": "/usr/local/bin/pnpm",
            "wine": "/usr/bin/wine",
        }.get
        tools = services_linux.resolve_tool_paths("/home/dev/.mt5", which=fake_which)
        self.assertEqual(tools.uv_bin, "/usr/local/bin/uv")
        self.assertEqual(tools.pnpm_bin, "/usr/local/bin/pnpm")
        self.assertEqual(tools.wine_bin, "/usr/bin/wine")
        self.assertIn("/home/dev/.mt5/drive_c/users/", tools.wine_python)
        self.assertTrue(tools.wine_python.endswith("Python312/python.exe"))

    def test_missing_tool_raises_with_a_clear_message(self) -> None:
        fake_which = {"uv": "/usr/local/bin/uv", "pnpm": None, "wine": None}.get
        with self.assertRaises(services_linux.MissingToolError) as ctx:
            services_linux.resolve_tool_paths("/home/dev/.mt5", which=fake_which)
        self.assertIn("pnpm", str(ctx.exception))
        self.assertIn("wine", str(ctx.exception))
        self.assertNotIn("uv,", str(ctx.exception))  # uv was found, shouldn't be listed as missing


class RenderAllTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rendered = services_linux.render_all(
            _fake_state(), FAKE_TOOLS, env_values=FAKE_ENV_VALUES
        )

    def test_backend_unit(self) -> None:
        text = self.rendered[services_linux.BACKEND_UNIT_NAME]
        self.assertIn("Restart=on-failure", text)
        self.assertIn("RestartSec=5", text)
        self.assertIn("WorkingDirectory=/opt/trading-bot/backend", text)
        self.assertIn("ExecStart=/fake/bin/uv run uvicorn src.main:socket_app --port 8000", text)

    def test_frontend_unit_uses_port_from_env(self) -> None:
        text = self.rendered[services_linux.FRONTEND_UNIT_NAME]
        self.assertIn("Restart=on-failure", text)
        self.assertIn("WorkingDirectory=/opt/trading-bot/frontend", text)
        self.assertIn("ExecStart=/fake/bin/pnpm dev --port 3001", text)

    def test_frontend_unit_falls_back_to_default_port_without_env(self) -> None:
        rendered = services_linux.render_all(_fake_state(), FAKE_TOOLS, env_values={})
        text = rendered[services_linux.FRONTEND_UNIT_NAME]
        self.assertIn(f"--port {services_linux.DEFAULT_FRONTEND_PORT}", text)

    def test_gateway_unit_named_and_instantiated_per_account(self) -> None:
        self.assertIn("trading-bot-gateway@default.service", self.rendered)
        self.assertIn("trading-bot-gateway@ftmo-1.service", self.rendered)

    def test_gateway_unit_carries_account_specific_values(self) -> None:
        default_text = self.rendered["trading-bot-gateway@default.service"]
        self.assertIn("Environment=GATEWAY_HOST=127.0.0.1", default_text)
        self.assertIn("Environment=GATEWAY_PORT=8787", default_text)
        self.assertIn("Environment=GATEWAY_SHARED_SECRET=primary-secret-value", default_text)
        self.assertIn("Restart=on-failure", default_text)

        ftmo_text = self.rendered["trading-bot-gateway@ftmo-1.service"]
        self.assertIn("Environment=GATEWAY_PORT=8788", ftmo_text)
        self.assertIn("Environment=GATEWAY_SHARED_SECRET=ftmo-secret-value", ftmo_text)

    def test_gateway_unit_carries_resolved_terminal_path(self) -> None:
        # Regression: the gateway unit used to hardcode an empty
        # MT5_TERMINAL_SUBPATH, so mt5_client.py never learned which
        # terminal to attach to and silently fell back to "whatever
        # terminal is already running" -- exactly the cross-account bug
        # mt5_terminal_path/mt5_terminal_subpath exists to prevent. Every
        # gateway unit must carry its own account's resolved terminal path,
        # explicit or defaulted, never blank.
        default_text = self.rendered["trading-bot-gateway@default.service"]
        default_terminal_path = "/home/dev/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"
        self.assertIn(f"Environment=MT5_TERMINAL_PATH={default_terminal_path}", default_text)

        # ftmo-1's terminal_path is None in the fixture -- must still resolve
        # to the wine-prefix default, not an empty/missing env line.
        ftmo_text = self.rendered["trading-bot-gateway@ftmo-1.service"]
        self.assertIn(
            "Environment=MT5_TERMINAL_PATH="
            + services_linux.default_terminal_path("/home/dev/.mt5"),
            ftmo_text,
        )
        self.assertNotIn("Environment=MT5_TERMINAL_SUBPATH=", ftmo_text)

    def test_gateway_unit_leaves_percent_i_for_systemd_to_resolve(self) -> None:
        text = self.rendered["trading-bot-gateway@ftmo-1.service"]
        self.assertIn("Requires=trading-bot-terminal@%i.service", text)
        self.assertIn('for account "%i"', text)

    def test_terminal_unit_uses_configured_terminal_path(self) -> None:
        text = self.rendered["trading-bot-terminal@default.service"]
        self.assertIn(
            'ExecStart=/fake/bin/wine '
            '"/home/dev/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"',
            text,
        )
        self.assertIn("Restart=on-failure", text)

    def test_terminal_unit_falls_back_to_default_path_when_unset(self) -> None:
        text = self.rendered["trading-bot-terminal@ftmo-1.service"]
        self.assertIn(
            'ExecStart=/fake/bin/wine '
            '"/home/dev/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"',
            text,
        )

    def test_target_wants_every_generated_unit(self) -> None:
        text = self.rendered[services_linux.TARGET_NAME]
        for name in (
            services_linux.BACKEND_UNIT_NAME,
            services_linux.FRONTEND_UNIT_NAME,
            "trading-bot-gateway@default.service",
            "trading-bot-terminal@default.service",
            "trading-bot-gateway@ftmo-1.service",
            "trading-bot-terminal@ftmo-1.service",
        ):
            self.assertIn(name, text)

    def test_no_account_secret_leaks_across_accounts(self) -> None:
        default_text = self.rendered["trading-bot-gateway@default.service"]
        self.assertNotIn("ftmo-secret-value", default_text)
        ftmo_text = self.rendered["trading-bot-gateway@ftmo-1.service"]
        self.assertNotIn("primary-secret-value", ftmo_text)


class GenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.unit_dir = Path(self._tmp.name) / "systemd-user"
        self._patcher = mock.patch.object(
            services_linux, "resolve_tool_paths", return_value=FAKE_TOOLS
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._tmp.cleanup()

    def test_dry_run_writes_nothing_to_disk(self) -> None:
        rendered = services_linux.generate(_fake_state(), dry_run=True, unit_dir=self.unit_dir)
        self.assertFalse(self.unit_dir.exists())
        self.assertIn(services_linux.TARGET_NAME, rendered)

    def test_real_run_writes_every_unit_file(self) -> None:
        rendered = services_linux.generate(_fake_state(), dry_run=False, unit_dir=self.unit_dir)
        for filename in rendered:
            self.assertTrue((self.unit_dir / filename).exists(), f"missing {filename}")

    def test_gateway_unit_files_are_written_owner_only(self) -> None:
        services_linux.generate(_fake_state(), dry_run=False, unit_dir=self.unit_dir)
        gateway_file = self.unit_dir / "trading-bot-gateway@default.service"
        mode = stat.S_IMODE(gateway_file.stat().st_mode)
        self.assertEqual(mode, 0o600)

    def test_non_gateway_unit_files_are_not_forced_owner_only(self) -> None:
        services_linux.generate(_fake_state(), dry_run=False, unit_dir=self.unit_dir)
        backend_file = self.unit_dir / services_linux.BACKEND_UNIT_NAME
        mode = stat.S_IMODE(backend_file.stat().st_mode)
        self.assertNotEqual(mode, 0o600)


class InstallUninstallDryRunTests(unittest.TestCase):
    """install()/uninstall() must never call systemctl/loginctl for real in
    these tests -- dry_run=True short-circuits _run() before any
    subprocess.run() call, which is asserted directly here rather than
    trusted blindly."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.unit_dir = Path(self._tmp.name) / "systemd-user"
        self._tools_patcher = mock.patch.object(
            services_linux, "resolve_tool_paths", return_value=FAKE_TOOLS
        )
        self._tools_patcher.start()
        self._subprocess_patcher = mock.patch.object(services_linux.subprocess, "run")
        self.mock_subprocess_run = self._subprocess_patcher.start()

    def tearDown(self) -> None:
        self._subprocess_patcher.stop()
        self._tools_patcher.stop()
        self._tmp.cleanup()

    def test_install_dry_run_never_shells_out(self) -> None:
        services_linux.install(_fake_state(), dry_run=True, unit_dir=self.unit_dir)
        self.mock_subprocess_run.assert_not_called()
        self.assertFalse(self.unit_dir.exists())

    def test_uninstall_dry_run_never_shells_out_and_does_not_delete(self) -> None:
        services_linux.generate(_fake_state(), dry_run=False, unit_dir=self.unit_dir)
        self.mock_subprocess_run.reset_mock()

        services_linux.uninstall(dry_run=True, unit_dir=self.unit_dir)

        self.mock_subprocess_run.assert_not_called()
        self.assertTrue((self.unit_dir / services_linux.TARGET_NAME).exists())

    def test_uninstall_real_run_removes_generated_files_via_mocked_systemctl(self) -> None:
        services_linux.generate(_fake_state(), dry_run=False, unit_dir=self.unit_dir)
        self.mock_subprocess_run.reset_mock()

        services_linux.uninstall(dry_run=False, unit_dir=self.unit_dir)

        # every trading-bot-* unit file this test generated should be gone
        remaining = list(self.unit_dir.glob("trading-bot*"))
        self.assertEqual(remaining, [])
        # systemctl was "called" only against the mock, never for real
        self.assertTrue(self.mock_subprocess_run.called)
        for call in self.mock_subprocess_run.call_args_list:
            cmd = call.args[0]
            self.assertEqual(cmd[0], "systemctl")


if __name__ == "__main__":
    unittest.main()
