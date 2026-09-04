import sqlite3
import datetime

conn = sqlite3.connect("backend/data/trading.db")
c = conn.cursor()
c.execute("SELECT id, symbol, open_time FROM trades ORDER BY open_time DESC LIMIT 5")
trades = c.fetchall()
for t in trades:
    open_time = t[2]
    # Local time
    local_dt = datetime.datetime.fromtimestamp(open_time)
    # Broker time (UTC for deriv)
    utc_dt = datetime.datetime.fromtimestamp(open_time, datetime.timezone.utc)
    print(f"Trade {t[0]}: DB Epoch={open_time} | Local={local_dt.strftime('%Y-%m-%d %H:%M:%S')} | Broker(UTC)={utc_dt.strftime('%Y-%m-%d %H:%M:%S')}")
