"""Interactive setup wizard: prompts the user through installer/install.py's
six steps and hands back a finished `WizardResult` for install.py to apply.

Every prompt has a sensible default (Enter accepts it). In `--non-interactive`
mode no prompt is actually shown — every question just returns its default.
In `--reconfigure` mode, defaults are pre-filled from `installer/state.json`
(the previous run's answers) and from what's already in `.env` /
`configs/accounts.yaml`, instead of the hardcoded defaults below.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

import accounts_writer
import env_writer
import wine_setup

DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 8787


@dataclass
class Account:
    id: str
    label: str
    mode: str  # "paper" | "live"
    gateway_host: str
    gateway_port: int
    terminal_path: str | None = None
    gateway_secret_env: str = ""

    @property
    def gateway_url(self) -> str:
        return f"http://{self.gateway_host}:{self.gateway_port}"


@dataclass
class Answers:
    install_dir: Path
    accounts: list[Account] = field(default_factory=list)
    wine_prefix: str | None = None
    run_wine_setup: bool = False
    autostart_on_boot: bool = True


@dataclass
class WizardResult:
    answers: Answers
    env_changes: list[env_writer.EnvChange]
    account_changes: list[accounts_writer.AccountChange]
    wine_commands: list[str]
    proceed: bool


def answers_to_state(answers: Answers) -> dict:
    """Serialize `answers` for `installer/state.json` — no secret *values*
    are ever produced by the wizard (only env var *names*), so this is safe
    to write in the clear."""
    return {
        "install_dir": str(answers.install_dir),
        "wine_prefix": answers.wine_prefix,
        "run_wine_setup": answers.run_wine_setup,
        "autostart_on_boot": answers.autostart_on_boot,
        "accounts": [
            {
                "id": a.id,
                "label": a.label,
                "mode": a.mode,
                "gateway_host": a.gateway_host,
                "gateway_port": a.gateway_port,
                "terminal_path": a.terminal_path,
                "gateway_secret_env": a.gateway_secret_env,
            }
            for a in answers.accounts
        ],
    }


def _ask(question: str, default: str = "", *, non_interactive: bool) -> str:
    if non_interactive:
        return default
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{question}{suffix}: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def _ask_yes_no(question: str, default: bool, *, non_interactive: bool) -> bool:
    if non_interactive:
        return default
    marker = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"{question} {marker}: ").strip().lower()
    except EOFError:
        raw = ""
    if not raw:
        return default
    return raw in ("y", "yes")


def _next_free_port(used_ports: set[int], start: int) -> int:
    port = start
    while port in used_ports:
        port += 1
    return port


def run(
    repo_root: Path,
    *,
    non_interactive: bool,
    reconfigure: bool,
    dry_run: bool,
    prior_state: dict | None = None,
) -> WizardResult:
    prior_state = prior_state or {}

    print("=" * 70)
    print("AI Trading Bot -- setup wizard")
    print("=" * 70)

    # ── Step 1: install directory ───────────────────────────────────────
    default_install_dir = prior_state.get("install_dir") or str(repo_root)
    install_dir_raw = _ask(
        "Install directory", default_install_dir, non_interactive=non_interactive
    )
    install_dir = Path(install_dir_raw).expanduser().resolve()

    accounts_path = install_dir / "configs" / "accounts.yaml"
    env_example_path = repo_root / ".env.example"

    existing_account_ids = accounts_writer.existing_ids(accounts_path)
    existing_ports = set(accounts_writer.existing_gateway_ports(accounts_path))

    # ── Step 2: accounts loop ───────────────────────────────────────────
    prior_accounts = prior_state.get("accounts") or []
    accounts: list[Account] = []
    used_ports = set(existing_ports)
    idx = 0
    # Whether configs/accounts.yaml already had any accounts *before* this
    # session -- true for essentially every real clone, since the file ships
    # committed with a `default` entry. Only when the file truly has zero
    # accounts does the first account added here inherit the plain
    # TB_GATEWAY_SHARED_SECRET var (matching a from-scratch primary account);
    # otherwise every account added in this session gets its own
    # TB_GATEWAY_SHARED_SECRET_<ID> -- even if it's the first one added in
    # *this* run -- so a custom id never ends up sharing a secret var with
    # the pre-existing primary account.
    file_has_existing_accounts = bool(existing_account_ids)
    if file_has_existing_accounts:
        print(
            "\nNote: every account added here gets its own "
            "TB_GATEWAY_SHARED_SECRET_<ID> in .env, since configs/accounts.yaml "
            "already has a primary account. Enter 'default' as the id if you're "
            "just configuring the existing primary account (it will be left "
            "untouched, with a warning, rather than duplicated)."
        )
    else:
        print(
            "\nNote: the first account added here uses the shared "
            f"{env_writer.PRIMARY_SECRET_VAR} (this is a from-scratch "
            "configs/accounts.yaml with no primary account yet). Every "
            "account after the first gets its own TB_GATEWAY_SHARED_SECRET_<ID>."
        )
    while True:
        is_first = idx == 0
        is_true_primary = is_first and not file_has_existing_accounts
        prompt_text = "Add an MT5 account?" if is_first else "Add another MT5 account?"
        default_add = bool(is_first)
        if idx < len(prior_accounts):
            default_add = True  # reconfigure: keep offering previously-configured slots

        if not _ask_yes_no(prompt_text, default_add, non_interactive=non_interactive):
            break

        prior_account = prior_accounts[idx] if idx < len(prior_accounts) else {}
        default_id = prior_account.get("id") or (
            "default" if is_true_primary else f"account-{idx + 1}"
        )
        account_id = (
            _ask("  Account id (slug)", default_id, non_interactive=non_interactive).strip()
            or default_id
        )

        if account_id in existing_account_ids or account_id in (a.id for a in accounts):
            print(
                f"  warning: '{account_id}' already exists in accounts.yaml -- it will be "
                "skipped when writing (not duplicated or modified)."
            )

        default_label = prior_account.get("label") or (
            "Primary MT5 account" if is_true_primary else f"MT5 account {idx + 1}"
        )
        label = _ask("  Label", default_label, non_interactive=non_interactive)

        default_mode = prior_account.get("mode") or "paper"
        mode = (
            _ask("  Mode (paper/live)", default_mode, non_interactive=non_interactive)
            .strip()
            .lower()
        )
        if mode not in ("paper", "live"):
            print(f"  '{mode}' isn't paper/live -- defaulting to paper.")
            mode = "paper"

        default_host = prior_account.get("gateway_host") or DEFAULT_GATEWAY_HOST
        host = _ask("  Gateway host", default_host, non_interactive=non_interactive)

        suggested_port = _next_free_port(
            used_ports, prior_account.get("gateway_port") or (DEFAULT_GATEWAY_PORT + idx)
        )
        port_raw = _ask("  Gateway port", str(suggested_port), non_interactive=non_interactive)
        try:
            port = int(port_raw)
        except ValueError:
            port = suggested_port
        used_ports.add(port)

        default_terminal = prior_account.get("terminal_path") or ""
        terminal_path = _ask(
            "  Path to this account's terminal64.exe (blank = attach to whatever "
            "terminal is already running)",
            default_terminal,
            non_interactive=non_interactive,
        ).strip()
        if terminal_path:
            if not Path(terminal_path).expanduser().exists():
                print(
                    f"  warning: '{terminal_path}' doesn't exist yet -- continuing "
                    "anyway (e.g. it'll be installed later)."
                )
        else:
            terminal_path = None

        secret_env = (
            env_writer.PRIMARY_SECRET_VAR
            if is_true_primary
            else env_writer.account_secret_var(account_id)
        )

        accounts.append(
            Account(
                id=account_id,
                label=label,
                mode=mode,
                gateway_host=host,
                gateway_port=port,
                terminal_path=terminal_path,
                gateway_secret_env=secret_env,
            )
        )
        idx += 1

    # ── Step 4: Wine + MT5 terminal auto-provisioning (Linux only) ─────
    run_wine_setup = False
    wine_prefix: str | None = None
    needs_terminal = any(a.terminal_path is None for a in accounts)
    if platform.system() == "Linux" and accounts and needs_terminal:
        run_wine_setup = _ask_yes_no(
            "\nSet up Wine + MT5 terminal automatically? (installs system packages "
            "via apt/winetricks)",
            False,
            non_interactive=non_interactive,
        )
        if run_wine_setup:
            default_wine_prefix = prior_state.get("wine_prefix") or str(Path.home() / ".mt5")
            wine_prefix = _ask(
                "  Wine prefix directory", default_wine_prefix, non_interactive=non_interactive
            )

    # ── Step 5: autostart intent ────────────────────────────────────────
    autostart_default = prior_state.get("autostart_on_boot", True)
    if platform.system() == "Linux":
        autostart_prompt = (
            "\nAutostart on boot? (registers systemd --user services for backend, frontend, "
            "and every account's gateway+terminal, enabled --now, plus loginctl enable-linger "
            "so they survive logout/reboot)"
        )
    else:
        autostart_prompt = (
            "\nAutostart on boot? (this only records your intent into installer/state.json -- "
            "Windows Scheduled Task registration needs a separate, explicit PowerShell step "
            "printed at the end, since it isn't safe to auto-elevate from here)"
        )
    autostart_on_boot = _ask_yes_no(
        autostart_prompt,
        autostart_default,
        non_interactive=non_interactive,
    )

    answers = Answers(
        install_dir=install_dir,
        accounts=accounts,
        wine_prefix=wine_prefix,
        run_wine_setup=run_wine_setup,
        autostart_on_boot=autostart_on_boot,
    )

    env_changes = env_writer.plan(install_dir, env_example_path, accounts, wine_prefix)
    account_changes = accounts_writer.plan(accounts_path, accounts)
    wine_commands = wine_setup.describe(Path(wine_prefix)) if run_wine_setup and wine_prefix else []

    # ── Step 6: final confirmation ──────────────────────────────────────
    print("\n" + "-" * 70)
    print("The following will happen:")
    if env_changes:
        for c in env_changes:
            print(f"  {c.description}")
    else:
        print("  .env: no changes")
    if account_changes:
        for c in account_changes:
            if c.skipped:
                print(f"  configs/accounts.yaml: warning -- {c.skip_reason}")
            else:
                print(f"  {c.description}")
    else:
        print("  configs/accounts.yaml: no changes")
    if wine_commands:
        print("  Wine provisioning commands (sudo required):")
        for cmd in wine_commands:
            print(f"    $ {cmd}")
    if autostart_on_boot and platform.system() == "Linux":
        print(
            "  autostart: register + enable --now systemd --user services, "
            "plus loginctl enable-linger"
        )
    elif autostart_on_boot:
        print(
            "  autostart: intent recorded (installer/state.json) -- run "
            "installer/services/windows/install_services.ps1 yourself next, "
            "printed at the end"
        )
    else:
        print("  autostart: no service registration (autostart declined)")
    print("-" * 70)

    if dry_run:
        proceed = False
    elif non_interactive:
        print("--non-interactive: proceeding without a confirmation prompt.")
        proceed = True
    else:
        proceed = _ask_yes_no("Proceed?", False, non_interactive=False)

    return WizardResult(
        answers=answers,
        env_changes=env_changes,
        account_changes=account_changes,
        wine_commands=wine_commands,
        proceed=proceed,
    )
