> 🇫🇷 Version française : [README.fr.md](README.fr.md)

# MT5 Gateway

The only component that talks to MetaTrader 5. It wraps the official
`MetaTrader5` Python package (Windows-only) behind a small FastAPI HTTP API so
the rest of the system can run natively on Linux.

**No business logic lives here** — the gateway reports broker facts (candles,
ticks, spread, positions) and executes explicit order commands. All decisions
happen in `backend/`.

> Setting this up on a machine other than your own dev checkout? Everything
> below this point — Wine provisioning, terminal configuration, the
> Makefile's per-account gateway commands — has a scripted equivalent in
> `installer/install.py` (see the repo root `README.md`'s "Installing on
> another machine"). This file stays the full manual reference: what the
> installer actually runs under the hood, and what to read when something
> needs troubleshooting by hand.

## Why it exists

The `MetaTrader5` pip package requires a running MT5 **desktop terminal** and
only works on Windows. The gateway isolates that constraint behind HTTP, so
deployment is a detail, not a code problem.

## Choosing where to run it

| Option | Use for | Verdict |
|--------|---------|---------|
| **A. Wine on Linux** | Development and paper trading on this machine | ✅ Fine for Phases 1–8 |
| **B. Windows VPS** | 24/7 live trading | ✅ **Recommended for live** (Phase 9) |

**Recommendation:** develop with **Wine** (free, local, good enough for paper
mode), and move to a **Windows VPS near your broker's servers** before any
live trading. A VPS gives you low latency, no sleep/suspend, no Wine quirks
mid-trade, and survives your laptop being off — all things that matter once
real money and open positions are involved.

## Prerequisites (both options)

- A broker **demo account** (login number, password, server name — e.g.
  `MetaQuotes-Demo`). Stay on demo until Phase 9's go-live criteria are met.
- Your broker's MT5 installer (`mt5setup.exe`), or the generic one from
  metatrader5.com.
- Python **3.12+ for Windows** (the *Windows* build, even under Wine — the
  `MetaTrader5` package only ships Windows wheels).

## Option A — Wine (development)

1. Install Wine (Ubuntu/Debian):
   ```bash
   sudo dpkg --add-architecture i386
   sudo apt update && sudo apt install --install-recommends wine64 wine32 winetricks
   ```
2. Install the MT5 terminal into a dedicated prefix:
   ```bash
   export WINEPREFIX=~/.mt5
   winetricks -q corefonts        # readable terminal UI
   wine mt5setup.exe
   ```
3. Install **Windows** Python 3.12 into the same prefix:
   ```bash
   wine python-3.12.x-amd64.exe /quiet InstallAllUsers=0 PrependPath=1
   ```
4. Install the gateway dependencies with Windows pip and start the gateway:
   ```bash
   wine python -m pip install MetaTrader5 fastapi uvicorn pydantic
   cd gateway
   wine python run_gateway.py
   ```
   The `run_gateway.py` launcher adds `src/` to `sys.path` automatically —
   this is needed because Wine does not pass `PYTHONPATH` through to the
   Windows Python process.
5. Start the MT5 terminal in the same prefix and leave it running:
   ```bash
   WINEPREFIX=~/.mt5 wine "$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe" &
   ```
   Then configure the terminal — see [Terminal configuration](#terminal-configuration-both-options).

Wine caveats worth knowing:

- Terminal and gateway must run in the **same Wine prefix** — the Python
  package finds the terminal through it.
- **Wine does not forward `PYTHONPATH`** to Windows processes — always use
  `run_gateway.py` (which injects `src/` into `sys.path`) instead of calling
  `uvicorn` directly.
- If the terminal renders a black/blank window, try
  `winetricks -q dxvk` or run with `wine explorer /desktop=mt5,1600x900 terminal64.exe`.
- Disable system suspend while paper trading overnight; a suspended terminal
  is a disconnected terminal.

## Option B — Windows VPS (recommended for live)

1. Rent a small Windows Server VPS **near your broker's trade servers**
   (brokers publish their datacenter locations; ping the server name shown in
   the MT5 login dialog and pick the region with the lowest RTT).
2. Install the MT5 terminal and Python 3.12, then:
   ```powershell
   pip install MetaTrader5 fastapi uvicorn pydantic
   ```
3. Copy the `gateway/` folder to the VPS and run it as a service so it
   survives reboots — e.g. with [NSSM](https://nssm.cc):
   ```powershell
   nssm install mt5-gateway "C:\Python312\python.exe" ^
     "C:\trading-bot\gateway\run_gateway.py"
   nssm set mt5-gateway AppDirectory "C:\trading-bot\gateway"
   nssm set mt5-gateway AppEnvironmentExtra GATEWAY_SHARED_SECRET=<long-random-string>
   nssm start mt5-gateway
   ```
   Also add the MT5 terminal to startup (Task Scheduler → run `terminal64.exe`
   at logon, with auto-logon enabled) so a VPS reboot brings everything back.
4. **Never expose the gateway to the public internet.** Bind it to
   `127.0.0.1` and reach it from the backend over **WireGuard** or an SSH
   tunnel:
   ```bash
   ssh -N -L 8787:127.0.0.1:8787 user@your-vps   # backend then uses 127.0.0.1:8787
   ```
5. Configure the terminal — next section.

## Terminal configuration (both options)

Do this once inside the MT5 terminal, logged into your **demo** account:

1. **Log in**: File → *Login to Trade Account* → enter login / password /
   server. Tick *Save password* so the terminal reconnects after restarts.
   (The app UI login (F11) re-authenticates through the gateway anyway; the
   terminal being logged in keeps data flowing between backend restarts.)
2. **Enable algorithmic trading**: Tools → Options → *Expert Advisors* →
   check **Allow algorithmic trading**, and make sure the **Algo Trading**
   toolbar button is ON (green). Without it, order calls in Phase 3+ fail
   with retcode `10027` (client disabled algo-trading).
3. **Market Watch symbols**: right-click Market Watch → *Symbols* → make
   **XAUUSD, XAGUSD, BTCUSD** visible. The gateway calls `symbol_select`
   defensively, but do check the **exact names** your broker uses — some
   brokers suffix them (`XAUUSD.a`, `GOLDmicro`, `BTCUSD.x`). If they differ,
   update `configs/app.yaml` and `configs/symbols/*.yaml` accordingly.
4. **History depth**: Tools → Options → Charts → set *Max bars in chart* to
   the maximum. MT5 only serves history it has downloaded; open an M5/H1/H4/D1
   chart of each symbol once and scroll back to force a download before
   running `POST /market-data/backfill`.
5. **Keep it running**: the Python API only works while the terminal process
   is up and connected. The gateway's `/health` reports
   `terminal_connected: false` whenever it isn't, and the backend pauses
   streaming until it returns.

## Running more than one account

`MetaTrader5`'s Python package only supports one logged-in account per OS
process — so N broker accounts means N gateway processes, each with its own
port, terminal instance (a separate Wine prefix per account is simplest on
Linux; a VPS naturally gives you one terminal per account), and shared
secret. Everything above still applies per-instance; what changes is how you
launch and stop them.

Each account is an entry in the repo root's `configs/accounts.yaml`
(`gateway_url` fixes its port, `gateway_shared_secret_env` names the `.env`
variable holding its secret — a distinct one per account). From the repo
root:

```bash
make dev-gateway ACCOUNT=ftmo-1     # this one account's gateway (defaults to
                                     # the first enabled account if ACCOUNT is omitted)
make dev-gateway-all                # every enabled account's gateway at once
make kill ACCOUNT=ftmo-1            # stop just that one gateway
```

`make dev-gateway` writes `gateway/run/<account_id>.pid` for the duration of
the process — that's what lets `make kill` (or `make kill ACCOUNT=...`) stop
one gateway precisely instead of `pkill`-ing every `run_gateway.py` process
on the machine.

## Wiring the backend to the gateway

In the repo root `.env` (see `.env.example`):

```bash
TB_GATEWAY_URL=http://127.0.0.1:8787         # or the tunnel endpoint
TB_GATEWAY_SHARED_SECRET=<same value as GATEWAY_SHARED_SECRET on the gateway>
```

Every request except `/health` must carry the secret in the
`X-Gateway-Secret` header — the backend's adapters do this automatically. If
`GATEWAY_SHARED_SECRET` is unset on the gateway the check is skipped; only
acceptable when both processes share a machine and the port is bound to
localhost.

## Smoke test

```bash
# 1. Gateway up? (no secret needed)
curl -s http://127.0.0.1:8787/health
#    → {"status":"ok","terminal_connected":false,"account":null}

# 2. Login through the gateway (or just use the app UI's MT5 Account panel):
curl -s -X POST http://127.0.0.1:8787/login \
  -H "X-Gateway-Secret: $TB_GATEWAY_SHARED_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"login": 12345678, "password": "...", "server": "MetaQuotes-Demo"}'

# 3. Candles flowing?
curl -s "http://127.0.0.1:8787/candles?symbol=XAUUSD&timeframe=M5&count=3" \
  -H "X-Gateway-Secret: $TB_GATEWAY_SHARED_SECRET"
```

Then start the backend (`make dev-backend`) and connect from the UI's
**MT5 Account** panel — status turns green and the candle stream begins
logging `candle closed …` lines on M5 boundaries.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `502 login rejected: [-6] Authorization failed` | Wrong login/password/**server name** (copy it exactly from the broker email) |
| `502 not logged in — POST /login first` | Backend not connected yet — use the UI login panel |
| `terminal_connected: false` after login worked before | Terminal closed/crashed or lost broker connection — restart it, gateway reconnects on next `/login` |
| `/candles` returns very few bars | Terminal hasn't downloaded that history — open the chart & scroll back (see History depth above) |
| `401 bad or missing X-Gateway-Secret` | `TB_GATEWAY_SHARED_SECRET` (backend) ≠ `GATEWAY_SHARED_SECRET` (gateway) |
| Order calls fail with retcode `10027` (Phase 3+) | Algo Trading disabled in the terminal — see Terminal configuration step 2 |
| Order calls fail with retcode `10030` ("unsupported filling mode") | The gateway asks the symbol which fill mode it supports (`symbol_info().filling_mode`) and picks FOK/IOC/RETURN accordingly — it should never hardcode one. If you still hit this, the symbol's broker-reported filling mode itself is wrong/missing; check the symbol's Trading tab in MT5's Symbols window |
| Wine terminal loses connection when laptop sleeps | Disable suspend, or move to the VPS option |

## API (implemented in Phase 1)

| Endpoint | Purpose |
|----------|---------|
| `POST /login` | Connect to an MT5 account (credentials held in memory only) |
| `POST /logout` | Shut down the terminal connection |
| `GET /health` | Terminal & account connection status (no auth) |
| `GET /candles` | OHLCV history (symbol, timeframe M5/H1/H4/D1, count) |
| `GET /tick` | Latest bid/ask |
| `GET /symbol_info` | Contract specs + live spread + stops level |
| `POST /order` | Open/modify an order *(Phase 3)* |
| `POST /close` | Close a position *(Phase 3)* |
| `GET /positions` | Open positions *(Phase 3)* |
