"""
Snapshot de cierre de la cadena SPX/SPXW de CBOE (2-60 DTE, +-10% del spot) en snapshots/.

Se toma DESPUES del cierre de EE.UU., cuando la ventana de captura es amplia:
  - el OI no cambia durante el dia ni al terminar la sesion (se actualiza de madrugada);
  - el volumen ya es el del dia completo (las cotizaciones de CBOE llegan con ~15 min de retraso).
Asi un solo archivo por dia trae el OI de ese dia y el volumen de ese mismo dia, y no importa
que GitHub retrase la ejecucion programada unos minutos u horas.

El cambio de OI entre el cierre de ayer y el de hoy corresponde a lo operado ayer, y se
compara con el volumen del archivo de ayer (lo hace detector.py).

No guarda (y sale sin error) cuando:
  - la ultima sesion regular de los datos no es la de hoy (feriado o feed sin actualizar);
  - aun no pasan las 16:40 ET (datos incompletos) o ya pasaron las 18:30 ET (el OI podria
    haber cambiado);
  - ya existe el archivo de esa sesion;
  - el OI casi no cambio respecto al cierre anterior.
Falla (exit 1) solo si no logra descargar la cadena.

FORCE=1 omite esas comprobaciones y guarda <fecha>_manualHHMM.csv.gz (el detector lo ignora).
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
EARLIEST, LATEST = 16 * 60 + 40, 18 * 60 + 30      # minutos, hora de Nueva York


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
    """Fecha de la ultima sesion regular (9:30-16:15 hora de Nueva York).

    Se ignoran las operaciones nocturnas (GTH): el SPX negocia casi las 24 h.
    """
    t = pd.to_datetime(last_trade_time, errors="coerce").dropna()
    hm = t.dt.hour * 60 + t.dt.minute
    t = t[(hm >= 9 * 60 + 30) & (hm <= 16 * 60 + 15)]
    return t.dt.date.max() if len(t) else None


def main(now=None, fetch_fn=fetch, force=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    force = (os.getenv("FORCE") == "1") if force is None else force
    ny = now.astimezone(ZoneInfo("America/New_York"))
    mins = ny.hour * 60 + ny.minute

    j = fetch_fn()
    d = j["data"]
    spot = d["current_price"]
    df = pd.DataFrame(d["options"])
    session = session_of(df["last_trade_time"])
    if session is None:
        sys.exit("La cadena no trae ninguna hora de ultimo trade; respuesta inesperada de CBOE.")

    if not force:
        if session != ny.date():
            print(f"La ultima sesion regular en los datos es {session}, no la de hoy ({ny.date()}): "
                  "feriado, fin de semana o el feed aun no la refleja. No se guarda.")
            return None
        if mins < EARLIEST:
            print("Aun es temprano: faltan datos del cierre (las cotizaciones llevan ~15 min de "
                  "retraso). No se guarda; un respaldo programado lo reintentara.")
            return None
        if mins > LATEST:
            print("Es demasiado tarde (despues de las 18:30 ET): el OI podria haberse actualizado "
                  "y mezclaria dias. No se guarda.")
            return None

    m = df["option"].str.extract(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
    m.columns = ["root", "exp", "cp", "strike"]
    df = pd.concat([df, m], axis=1).dropna(subset=["exp"])
    df["exp"] = pd.to_datetime(df["exp"], format="%y%m%d")
    df["strike"] = df["strike"].astype(int) / 1000
    dte = (df["exp"] - pd.Timestamp(ny.date())).dt.days
    df = df[(dte >= 2) & (dte <= 60) & df["strike"].between(spot * 0.9, spot * 1.1)]
    df = df[KEEP].assign(session=str(session), cboe_ts=j["timestamp"], spot=spot,
                         captured_utc=now.isoformat())

    OUT.mkdir(exist_ok=True)
    path = OUT / (f"{ny.date()}_manual{ny:%H%M}.csv.gz" if force else f"{session}_pm.csv.gz")
    if not force:
        if path.exists():
            print(f"Ya existe {path.name}: no se guarda otra vez.")
            return None
        prevs = [f for f in sorted(OUT.glob("*_pm.csv.gz")) if f.name[:10] < str(session)]
        if prevs:
            prev = pd.read_csv(prevs[-1], usecols=["option", "open_interest"])
            both = prev.merge(df[["option", "open_interest"]], on="option", suffixes=("_prev", "_new"))
            changed = (both["open_interest_prev"] != both["open_interest_new"]).mean()
            if len(both) and changed < 0.05:
                print(f"El OI casi no cambio respecto a {prevs[-1].name} ({changed:.1%} de "
                      f"{len(both):,} contratos): CBOE no actualizo el OI hoy. No se guarda.")
                return None

    df.to_csv(path, index=False, compression="gzip")
    print(f"Guardado {path} | sesion {session} | {len(df):,} filas | {path.stat().st_size / 1e3:.0f} KB")
    return path


if __name__ == "__main__":
    main()
