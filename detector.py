"""
Detector de estructuras de varias patas en SPX/SPXW (sin 0DTE) con datos gratis de CBOE.

Compara dos snapshots (OI anterior vs OI actual), busca patas con aumento de OI
y las agrupa en verticales, butterflies, iron butterflies/condors y straddles/strangles.
NO determina si la estructura fue comprada o vendida: es una hipotesis, no un dato.

Uso:
  python detector.py --dir snapshots                       # elige solo los 2 snapshots
  python detector.py --prev snapshots/A.csv.gz --cur snapshots/B.csv.gz
  python detector.py --dir snapshots --closes              # busca cierres (OI baja)
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- carga
def pick_files(folder):
    files = sorted(Path(folder).glob("*.csv.gz"))
    if not files:
        sys.exit(f"No hay archivos .csv.gz en {folder}")
    cur = files[-1]
    older = [f for f in files if f.name[:10] < cur.name[:10]]
    if not older:
        sys.exit("Falta un snapshot de un dia anterior para calcular el cambio de OI.")
    return older[-1], cur


def load(path):
    df = pd.read_csv(path, parse_dates=["exp"])
    df["ltt"] = pd.to_datetime(df["last_trade_time"], errors="coerce")
    return df


def checks(prev, cur):
    cap_prev = pd.Timestamp(prev["captured_utc"].iloc[0]).tz_convert("UTC")
    cap_cur = pd.Timestamp(cur["captured_utc"].iloc[0]).tz_convert("UTC")
    gap = int(np.busday_count(cap_prev.date(), cap_cur.date()))
    if gap > 1:
        print(f"AVISO: el cambio de OI abarca {gap} dias habiles (faltan snapshots o hay feriado).")
    if cap_cur.hour + cap_cur.minute / 60 >= 13.5:
        print("AVISO: el snapshot actual se tomo durante o despues de la sesion; "
              "su OI aun no incluye las operaciones de ese dia. Usa uno de la manana.")
    return cap_cur


# ------------------------------------------------------------- patas
def build_legs(prev, cur, sign, a):
    m = prev[["option", "open_interest"]].merge(cur, on="option", suffixes=("_prev", ""))
    m["doi"] = sign * (m["open_interest"] - m["open_interest_prev"])
    m["mid"] = (m["bid"] + m["ask"]) / 2
    legs = m[(m["doi"] >= a.min_oi) & (m["volume"] > 0)].copy()
    return m, legs


# ---------------------------------------------------------- puntuacion
def score_structure(rows, weights, kind, session, a):
    s = rows["doi"].to_numpy(float)
    x = s / np.asarray(weights, float)
    dev = (x.max() - x.min()) / x.max()
    if dev > a.tol:
        return None
    sym = 1 - dev / a.tol
    unit = x.min()
    size = float(np.clip(np.log10(unit / a.min_oi) / np.log10(a.big / a.min_oi), 0, 1))
    vol = rows["volume"].to_numpy(float)
    conf = float(np.mean(np.minimum(s, vol) / np.maximum(s, vol)))
    # sincronia: fraccion de patas cuyo ultimo trade esta a <=60 s de otra pata (mismo dia de sesion)
    t = rows["ltt"]
    ok = t.notna() & (t.dt.date == session)
    flags = []
    for i in range(len(rows)):
        hit = False
        if ok.iloc[i]:
            for j in range(len(rows)):
                if j != i and ok.iloc[j] and abs((t.iloc[i] - t.iloc[j]).total_seconds()) <= 60:
                    hit = True
        flags.append(hit)
    sync = float(np.mean(flags))
    span = rows["strike"].max() - rows["strike"].min()
    width = float(np.clip(1 - (span - 150) / 250 * 0.7, 0.3, 1)) if span > 150 else 1.0
    total = 100 * (0.35 * size + 0.25 * sym + 0.20 * conf + 0.15 * sync + 0.05 * width)
    if kind in ("straddle", "strangle"):
        total *= 0.85
    return dict(score=round(total, 1), size=round(size, 2), sym=round(sym, 2),
                vol_conf=round(conf, 2), sync=round(sync, 2), unit=int(round(unit)))


# -------------------------------------------------------- estructuras
def find_structures(legs, session, a):
    cands = []
    for (root, exp), g in legs.groupby(["root", "exp"]):
        g = g.sort_values("doi", ascending=False).head(a.max_legs).sort_values("strike")
        calls = g[g["cp"] == "C"]
        puts = g[g["cp"] == "P"]

        def add(rows, weights, kind):
            res = score_structure(rows, weights, kind, session, a)
            if res and res["score"] >= a.min_score:
                cands.append(dict(kind=kind, root=root, exp=exp, rows=rows, **res))

        # verticales (2 patas, mismo tipo) y butterflies (3 patas 1:2:1)
        for side, name in ((calls, "call"), (puts, "put")):
            idx = list(side.index)
            for i, j in itertools.combinations(idx, 2):
                if abs(side.loc[j, "strike"] - side.loc[i, "strike"]) <= a.max_width:
                    add(side.loc[[i, j]], [1, 1], f"{name} spread")
            for i, j, k in itertools.combinations(idx, 3):
                r = side.loc[[i, j, k]]
                k1, k2, k3 = r["strike"].to_numpy()
                if abs((k2 - k1) - (k3 - k2)) <= 5 and (k3 - k1) <= a.max_width:
                    add(r, [1, 2, 1], f"{name} butterfly")

        # iron butterfly / iron condor (2 puts + 2 calls, mismo tamano)
        doi, stk = g["doi"].to_dict(), g["strike"].to_dict()

        def close(ids):
            v = [doi[i] for i in ids]
            return (max(v) - min(v)) / max(v) <= a.tol

        pp = [c for c in itertools.combinations(list(puts.index), 2) if close(c)]
        cc = [c for c in itertools.combinations(list(calls.index), 2) if close(c)]
        for p, c in itertools.product(pp, cc):
            if (stk[p[1]] <= stk[c[0]] and stk[p[1]] - stk[p[0]] <= a.max_width
                    and stk[c[1]] - stk[c[0]] <= a.max_width
                    and stk[c[0]] - stk[p[1]] <= 2 * a.max_width and close(p + c)):
                kind = "iron butterfly" if stk[p[1]] == stk[c[0]] else "iron condor"
                add(g.loc[list(p + c)], [1, 1, 1, 1], kind)

        # straddle / strangle (1 put + 1 call)
        for pi, ci in itertools.product(puts.index, calls.index):
            gap = calls.loc[ci, "strike"] - puts.loc[pi, "strike"]
            if 0 <= gap <= 2 * a.max_width:
                kind = "straddle" if puts.loc[pi, "strike"] == calls.loc[ci, "strike"] else "strangle"
                add(pd.concat([puts.loc[[pi]], calls.loc[[ci]]]), [1, 1], kind)
    return cands


def resolve(cands):
    """Cada pata se usa una sola vez; se prefieren estructuras con mas patas y mejor puntuacion."""
    cands = sorted(cands, key=lambda c: c["score"] + 6 * (len(c["rows"]) - 2), reverse=True)
    used, out = set(), []
    for c in cands:
        ids = set(c["rows"]["option"])
        if ids & used:
            continue
        used |= ids
        out.append(c)
    return out, used


# -------------------------------------------------------------- salida
def to_table(chosen, session_dte):
    rows = []
    for n, c in enumerate(sorted(chosen, key=lambda c: -c["score"]), 1):
        r = c["rows"].sort_values(["cp", "strike"], ascending=[False, True])
        rows.append(dict(
            id=n, estructura=c["kind"], root=c["root"], exp=c["exp"].date(),
            dte=(c["exp"] - session_dte).days,
            patas=" / ".join(f'{x.cp}{x.strike:g}' for x in r.itertuples()),
            doi_por_pata=" / ".join(f"{int(x):,}" for x in r["doi"]),
            contratos_unidad=c["unit"], score=c["score"], size=c["size"], simetria=c["sym"],
            conf_volumen=c["vol_conf"], sincronia=c["sync"],
        ))
    return pd.DataFrame(rows)


def make_chart(m, chosen, table, spot, title, path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly no esta instalado: se omite el grafico (pip install plotly).")
        return
    lo, hi = spot * 0.92, spot * 1.08
    d = m[m["strike"].between(lo, hi) & (m["doi"] != 0)]
    exps = sorted(d["exp"].unique())
    leg_rows = []
    for n, c in enumerate(sorted(chosen, key=lambda c: -c["score"]), 1):
        for x in c["rows"].itertuples():
            leg_rows.append(dict(id=n, exp=c["exp"], strike=x.strike, doi=x.doi, cp=x.cp,
                                 kind=c["kind"], score=c["score"], root=c["root"]))
    legs = pd.DataFrame(leg_rows, columns=["id", "exp", "strike", "doi", "cp", "kind", "score", "root"])

    fig = go.Figure()
    sets = [("Todos", None)] + [(str(pd.Timestamp(e).date()), e) for e in exps]
    for label, e in sets:
        sub = d if e is None else d[d["exp"] == e]
        ls = legs if e is None else legs[legs["exp"] == e]
        vis = label == "Todos"
        for cp, name, color in (("C", "Calls", "#5DA5DA"), ("P", "Puts", "#FAA43A")):
            z = sub[sub["cp"] == cp].groupby("strike")["doi"].sum()
            fig.add_trace(go.Bar(x=z.index, y=z.values, name=name, marker_color=color,
                                 visible=vis, legendgroup=name, showlegend=vis,
                                 hovertemplate="strike %{x}<br>ΔOI %{y:,.0f}<extra>" + name + "</extra>"))
        ls = ls.groupby(["id", "cp", "strike"], as_index=False).agg(
            doi=("doi", "sum"), kind=("kind", "first"), score=("score", "first"))
        fig.add_trace(go.Scatter(
            x=ls["strike"], y=ls["doi"], mode="markers+text", visible=vis, showlegend=False,
            text=["#" + str(i) for i in ls["id"]], textposition="top center",
            marker=dict(symbol="diamond", size=10, color="white", line=dict(width=1, color="black")),
            customdata=np.stack([ls["kind"], ls["score"]], axis=-1) if len(ls) else None,
            hovertemplate="#%{text} %{customdata[0]}<br>score %{customdata[1]}<br>"
                          "strike %{x}<br>ΔOI %{y:,.0f}<extra></extra>"))
    buttons = []
    for i, (label, _) in enumerate(sets):
        v = [False] * (3 * len(sets))
        v[3 * i:3 * i + 3] = [True, True, True]
        buttons.append(dict(label=label, method="update", args=[{"visible": v}]))
    fig.add_vline(x=spot, line_dash="dash", line_color="gray", annotation_text=f"spot {spot:,.0f}")
    fig.update_layout(
        template="plotly_dark", title=title, barmode="group", hovermode="closest",
        xaxis_title="Strike", yaxis_title="Cambio de OI (contratos)",
        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=1.12, xanchor="left")],
        legend=dict(orientation="h", y=-0.15),
        annotations=[dict(text="Fuente: CBOE (cotizaciones con retraso). El lado comprado/vendido no se conoce.",
                          xref="paper", yref="paper", x=0, y=-0.25, showarrow=False, font=dict(size=11))])
    fig.write_html(path)
    print("Grafico:", path)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="snapshots")
    p.add_argument("--prev")
    p.add_argument("--cur")
    p.add_argument("--out", default="detections")
    p.add_argument("--closes", action="store_true", help="buscar cierres (baja de OI)")
    p.add_argument("--min-oi", type=float, default=500, help="aumento minimo de OI por pata")
    p.add_argument("--big", type=float, default=4000, help="tamano que da puntuacion maxima")
    p.add_argument("--tol", type=float, default=0.25, help="diferencia relativa maxima entre patas")
    p.add_argument("--min-score", type=float, default=50)
    p.add_argument("--max-width", type=float, default=200,
                   help="ancho maximo (puntos) de un spread/ala; straddles y strangles hasta el doble")
    p.add_argument("--max-legs", type=int, default=40, help="patas maximas por vencimiento")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()

    pf, cf = (Path(a.prev), Path(a.cur)) if a.prev and a.cur else pick_files(a.dir)
    print(f"Anterior: {pf.name}\nActual:   {cf.name}")
    prev, cur = load(pf), load(cf)
    cap = checks(prev, cur)
    session = cur["ltt"].dropna().dt.date.max()      # dia al que pertenece el volumen
    spot = float(cur["spot"].iloc[0])

    sign = -1 if a.closes else 1
    m, legs = build_legs(prev, cur, sign, a)
    print(f"Contratos comparables: {len(m):,} | patas candidatas (ΔOI>={a.min_oi:g} y volumen>0): {len(legs):,}")

    chosen, used = resolve(find_structures(legs, session, a))
    table = to_table(chosen, pd.Timestamp(cap.date()))
    what = "cierres" if a.closes else "aperturas"
    print(f"\n{len(table)} estructuras ({what}) con score >= {a.min_score:g}. "
          "Lado comprado/vendido: desconocido.\n")
    if len(table):
        print(table.head(a.top).to_string(index=False))

    singles = legs[~legs["option"].isin(used)].sort_values("doi", ascending=False).head(15)
    if len(singles):
        print("\nPatas sueltas con mayor ΔOI (no encajan en ninguna estructura):")
        print(singles[["root", "exp", "cp", "strike", "open_interest_prev", "open_interest", "doi", "volume"]]
              .rename(columns={"open_interest_prev": "oi_ant", "open_interest": "oi_act", "doi": "d_oi"})
              .assign(exp=lambda d: d["exp"].dt.date).to_string(index=False))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cap.date()}{'_cierres' if a.closes else ''}"
    table.to_csv(out / f"structures_{tag}.csv", index=False)
    make_chart(m, chosen, table, spot,
               f"Cambio de OI SPX/SPXW: {pf.name[:10]} a {cf.name[:10]} ({what}; ◆ = estructuras detectadas)",
               str(out / f"chart_{tag}.html"))


if __name__ == "__main__":
    main()
