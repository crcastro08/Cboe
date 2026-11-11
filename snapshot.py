import datetime as dt, pathlib, pandas as pd, requests

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
r.raise_for_status()
j = r.json(); d = j["data"]
spot = d["current_price"]
cap = dt.datetime.now(dt.timezone.utc)

df = pd.DataFrame(d["options"])
m = df["option"].str.extract(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
m.columns = ["root", "exp", "cp", "strike"]
df = pd.concat([df, m], axis=1).dropna(subset=["exp"])
df["exp"] = pd.to_datetime(df["exp"], format="%y%m%d")
df["strike"] = df["strike"].astype(int) / 1000

dte = (df["exp"] - pd.Timestamp(cap.date())).dt.days
df = df[(dte >= 2) & (dte <= 60) & df["strike"].between(spot * 0.9, spot * 1.1)]

keep = ["option", "root", "exp", "cp", "strike", "bid", "ask", "iv",
        "open_interest", "volume", "delta", "gamma",
        "last_trade_price", "last_trade_time"]
df = df[keep].assign(cboe_ts=j["timestamp"], spot=spot, captured_utc=cap.isoformat())

tag = "am" if cap.hour < 18 else "pm"
out = pathlib.Path("snapshots"); out.mkdir(exist_ok=True)
path = out / f"{cap:%Y-%m-%d}_{tag}.csv.gz"
df.to_csv(path, index=False, compression="gzip")
print(path, len(df), "filas,", round(path.stat().st_size / 1e6, 2), "MB")
