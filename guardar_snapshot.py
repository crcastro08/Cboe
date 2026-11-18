"""
Guarda la cadena SPX/SPXW de CBOE en Supabase (PostgreSQL). Se ejecuta a mano.

Cuando correrlo: cada manana, despues de la actualizacion del OI y antes de la apertura de
EE.UU. (entre 8:00 y 9:25 de la manana, hora de Nueva York). Asi el OI ya incluye las
operaciones de la sesion anterior y el volumen es el de esa misma sesion completa.

Uso:
  python guardar_snapshot.py               guarda la sesion mas reciente
  python guardar_snapshot.py --prueba      descarga y muestra un resumen, sin guardar
  python guardar_snapshot.py --reemplazar  sobrescribe una sesion ya guardada
  python guardar_snapshot.py --forzar      omite las comprobaciones (no recomendado)

Requiere SUPABASE_DB_URL, en un archivo .env junto al script (ver .env.ejemplo).
"""
import argparse
import datetime as dt
import os
import pathlib
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg
import requests

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
NY = ZoneInfo("America/New_York")
CAMPOS = ["option", "bid", "ask", "iv", "open_interest", "volume", "delta", "gamma",
          "last_trade_price", "last_trade_time"]

INSERT_COTIZACION = """
    insert into cotizaciones (sesion, contrato_id, capturado_utc, spot, oi, volumen,
                              bid, ask, iv, delta, gamma, ultimo, ultimo_trade)
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def cargar_env():
    """Lee el archivo .env que esta junto al script (formato CLAVE=valor).

    Si la variable ya existe en el sistema, se respeta ese valor.
    """
    ruta = pathlib.Path(__file__).with_name(".env")
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


def descargar(intentos=3):
    for i in range(1, intentos + 1):
        try:
            r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                             timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Intento {i}/{intentos} fallo: {e}")
            if i < intentos:
                time.sleep(20)
    sys.exit("No se pudo descargar la cadena de CBOE.")


def sesion_de(opciones):
    """Ultima sesion regular (9:30-16:15 ET). Ignora operaciones nocturnas (GTH)."""
    t = pd.to_datetime(pd.Series([o.get("last_trade_time") for o in opciones]), errors="coerce").dropna()
    hm = t.dt.hour * 60 + t.dt.minute
    t = t[(hm >= 9 * 60 + 30) & (hm <= 16 * 60 + 15)]
    return t.dt.date.max() if len(t) else None


def parsear(j):
    d = j["data"]
    spot = float(d["current_price"])
    df = pd.DataFrame(d["options"])[CAMPOS]
    sym = df["option"].str.extract(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
    df = pd.concat([df, sym.set_axis(["raiz", "venc", "tipo", "k"], axis=1)], axis=1)
    df = df.dropna(subset=["venc"]).copy()
    df["vencimiento"] = pd.to_datetime(df["venc"], format="%y%m%d").dt.date
    df["strike"] = df["k"].astype(int) / 1000
    hoy = pd.Timestamp(dt.datetime.now(NY).date())
    dte = (pd.to_datetime(df["vencimiento"]) - hoy).dt.days
    df = df[(dte >= 2) & (dte <= 60) & df["strike"].between(spot * 0.9, spot * 1.1)].copy()
    return df, spot


def _f(x):
    return None if pd.isna(x) else float(x)


def _i(x):
    return None if pd.isna(x) else int(x)


def _ts(x):
    t = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(t) else t.to_pydatetime()


def guardar_contratos(cur, df):
    """Registra cada contrato una sola vez y devuelve el DataFrame con contrato_id."""
    cur.executemany(
        "insert into contratos (simbolo, raiz, vencimiento, tipo, strike) "
        "values (%s, %s, %s, %s, %s) on conflict (simbolo) do nothing",
        list(zip(df["option"].tolist(), df["raiz"].tolist(), df["vencimiento"].tolist(),
                 df["tipo"].tolist(), df["strike"].astype(float).tolist())))
    cur.execute("select simbolo, id from contratos where simbolo = any(%s)",
                (df["option"].tolist(),))
    mapa = dict(cur.fetchall())
    return df.assign(contrato_id=df["option"].map(mapa))


def filas(df, sesion, spot, capturado):
    return [(sesion, int(r.contrato_id), capturado, spot,
             _i(r.open_interest), _i(r.volume), _f(r.bid), _f(r.ask), _f(r.iv),
             _f(r.delta), _f(r.gamma), _f(r.last_trade_price), _ts(r.last_trade_time))
            for r in df.itertuples(index=False)]


def main():
    ap = argparse.ArgumentParser(description="Guarda la cadena de CBOE en Supabase.")
    ap.add_argument("--prueba", action="store_true", help="solo descargar y mostrar")
    ap.add_argument("--reemplazar", action="store_true", help="sobrescribir la sesion si ya existe")
    ap.add_argument("--forzar", action="store_true", help="omitir comprobaciones")
    a = ap.parse_args()
    cargar_env()

    j = descargar()
    sesion = sesion_de(j["data"]["options"])
    if sesion is None:
        sys.exit("CBOE no trae operaciones regulares: respuesta inesperada.")
    hoy = dt.datetime.now(NY).date()
    if sesion >= hoy and not (a.forzar or a.prueba):
        sys.exit(f"La sesion mas reciente es la de hoy ({sesion}): el OI todavia no incluye esas "
                 "operaciones. Vuelve manana antes de la apertura (8:00-9:25 ET).")

    df, spot = parsear(j)
    print(f"Sesion {sesion} | timestamp CBOE {j['timestamp']} | spot {spot:,.2f} | "
          f"contratos (2-60 DTE, +-10%): {len(df):,}")
    if a.prueba:
        print(df[["option", "open_interest", "volume", "last_trade_price"]].head(5).to_string(index=False))
        return

    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        sys.exit("Falta SUPABASE_DB_URL: crea el archivo .env junto al script (ver instrucciones).")

    capturado = dt.datetime.now(dt.timezone.utc)
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("select 1 from cotizaciones where sesion = %s limit 1", (sesion,))
        existe = cur.fetchone() is not None
        if existe and not a.reemplazar:
            print(f"La sesion {sesion} ya esta guardada. Usa --reemplazar para sobrescribirla.")
            return

        df = guardar_contratos(cur, df)

        cur.execute("select sesion from cotizaciones where sesion < %s order by sesion desc limit 1",
                    (sesion,))
        previa = cur.fetchone()
        if previa and not a.forzar:
            cur.execute("select contrato_id, oi from cotizaciones where sesion = %s", (previa[0],))
            prev = pd.DataFrame(cur.fetchall(), columns=["contrato_id", "oi_prev"])
            m = df[["contrato_id", "open_interest"]].merge(prev, on="contrato_id")
            cambio = (m["open_interest"] != m["oi_prev"]).mean() if len(m) else 0.0
            if cambio < 0.05:
                sys.exit(f"El OI casi no cambio frente a la sesion {previa[0]} ({cambio:.1%}). "
                         "CBOE aun no lo actualizo: intenta mas tarde (o usa --forzar).")

        if existe:
            cur.execute("delete from cotizaciones where sesion = %s", (sesion,))
        cur.executemany(INSERT_COTIZACION, filas(df, sesion, spot, capturado))

        cur.execute("select pg_size_pretty(pg_database_size(current_database()))")
        tamano = cur.fetchone()[0]

    print(f"Guardado: sesion {sesion}, {len(df):,} contratos. Tamano de la base: {tamano} (limite 500 MB).")


if __name__ == "__main__":
    main()