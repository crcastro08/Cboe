"""
Snapshot diario de la cadena SPX/SPXW de CBOE (2-60 DTE, +-10% del spot), guardado en snapshots/.

Pensado para correr por la manana antes de la apertura de EE.UU.: en ese momento el OI ya
incluye lo operado ayer y el volumen es el del dia anterior completo.

No guarda (y sale sin error) cuando:
  - la sesion de hoy ya empezo o termino (los datos serian parciales o el OI aun no cambia);
  - ya existe un snapshot de la manana para la misma sesion (feriados, ejecuciones de respaldo);
  - el OI no cambio respecto al ultimo snapshot de la manana (CBOE aun no lo actualiza).
Falla (exit 1) solo si no logra descargar la cadena.

FORCE=1 omite esas comprobaciones y guarda el archivo como <fecha>_manualHHMM.csv.gz.
"""
import datetime as dt
import os
import pathlib
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
OUT = pathlib.Path("snapshots")
KEEP = ["option", "root", "exp", "cp", "strike", "bid", "ask", "iv",
        "open_interest", "volume", "delta", "gamma", "last_trade_price", "last_trade_time"]


def fetch(retries=3):
    for i in range(1, retries + 1):
        try:
            r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                             timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:                      # red, HTTP o JSON invalido
            print(f"Intento {i}/{retries} fallo: {e}")
            if i < retries:
                time.sleep(30)
    sys.exit("No se pudo descargar la cadena de CBOE.")


def session_of(last_trade_time):
    """Fecha de la ultima sesion con operaciones (hora de Nueva York)."""
    t = pd.to_datetime(last_trade_time, errors="coerce").dropna()
    return t.dt.date.max() if len(t) else None


def main(now=None, fetch_fn=fetch, force=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    force = (os.getenv("FORCE") == "1") if force is None else force

    j = fetch_fn()
    d = j["data"]
    spot = d["current_price"]
    df = pd.DataFrame(d["options"])
    session = session_of(df["last_trade_time"])

    ny_today = now.astimezone(ZoneInfo("America/New_York")).date()
    if session is None:
        sys.exit("La cadena no trae ninguna hora de ultimo trade; respuesta inesperada de CBOE.")
    if session >= ny_today and not force:
        print(f"La sesion de hoy ({session}) ya empezo o termino: no se guarda. "
              "Este snapshot debe tomarse antes de la apertura.")
        return None

    m = df["option"].str.extract(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
    m.columns = ["root", "exp", "cp", "strike"]
    df = pd.concat([df, m], axis=1).dropna(subset=["exp"])
    df["exp"] = pd.to_datetime(df["exp"], format="%y%m%d")
    df["strike"] = df["strike"].astype(int) / 1000
    dte = (df["exp"] - pd.Timestamp(now.date())).dt.days
    df = df[(dte >= 2) & (dte <= 60) & df["strike"].between(spot * 0.9, spot * 1.1)]
    df = df[KEEP].assign(session=str(session), cboe_ts=j["timestamp"], spot=spot,
                         captured_utc=now.isoformat())

    OUT.mkdir(exist_ok=True)
    ams = sorted(OUT.glob("*_am.csv.gz"))
    if ams and not force:
        prev = pd.read_csv(ams[-1])
        prev_session = (str(prev["session"].iloc[0]) if "session" in prev
                        else str(session_of(prev["last_trade_time"])))
        if prev_session == str(session):
            print(f"Ya existe un snapshot de la manana para la sesion {session}: no se guarda.")
            return None
        both = prev[["option", "open_interest"]].merge(df[["option", "open_interest"]],
                                                       on="option", suffixes=("_prev", "_new"))
        changed = (both["open_interest_prev"] != both["open_interest_new"]).mean()
        if len(both) and changed < 0.05:
            print(f"El OI casi no cambio ({changed:.1%} de {len(both):,} contratos): CBOE aun no lo "
                  "actualiza. No se guarda; un respaldo programado lo reintentara.")
            return None

    tag = f"manual{now:%H%M}" if force else "am"
    path = OUT / f"{now:%Y-%m-%d}_{tag}.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    print(f"Guardado {path} | sesion {session} | {len(df):,} filas | {path.stat().st_size / 1e3:.0f} KB")
    return path


if __name__ == "__main__":
    main()
