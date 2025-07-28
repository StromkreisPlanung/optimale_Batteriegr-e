from __future__ import annotations
"""
streamlit_app.py · Battery‑Sizing Dashboard  v0.6.5
===================================================
• Mehrere PV‑CSV‑Dateien werden addiert
• Robuster CSV‑Reader (Semikolon/Komma + Dezimalpunkt/-komma)
• DST-Fix: Europe/Vienna korrekt behandeln, PyPSA bekommt tz‑naive Snapshots
• Smart‑EV‑Charging: index-sicher (kein reindex, sondern isin)
• Page‑Config‑Guard: Doppelaufrufe abgefangen
• PyPSA: p_set-Zeitreihen werden NACH dem Add gesetzt (keine Längenfehler)
"""

import io
from datetime import time

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import streamlit as st

# ------------------------------------------------------------- #
# 0) Page‑Config nur einmal setzen                               #
# ------------------------------------------------------------- #
try:
    st.set_page_config(page_title="Battery Sizing Tool", layout="wide")
except st.errors.StreamlitAPIException:
    # Page-Config wurde bereits gesetzt (z. B. durch Cloud)
    pass

# ------------------------------------------------------------- #
# 1) CSV‑Reader                                                  #
# ------------------------------------------------------------- #
def read_profile(upload, res: str) -> pd.Series:
    """
    Erwartetes CSV:
        datetime,power_kw        (oder Semikolon + Dezimalkomma)
        2025-01-01 00:15,123.4
    """
    raw = upload.getvalue().decode("utf-8-sig")
    header = raw.splitlines()[0] if raw else "datetime,power_kw"
    semicolon = ";" in header
    sep, dec = (";", ",") if semicolon else (",", ".")

    df = pd.read_csv(
        io.StringIO(raw),
        sep=sep,
        decimal=dec,
        names=["datetime", "power_kw"],
        header=0,
        parse_dates=["datetime"],
        dayfirst=semicolon,   # dt. Format 01.01.2025 …
        engine="python",
    )

    # Doppelte Zeitstempel vorher mitteln (z. B. bei Mehrfach-Uploads)
    df = df.groupby("datetime", as_index=False)["power_kw"].mean()

    # Index setzen und resamplen
    df.set_index("datetime", inplace=True)
    s = df["power_kw"].resample(res).mean().fillna(0.0)

    # 1) TZ korrekt anwenden (DST beachten) ...
    s.index = s.index.tz_localize(
        "Europe/Vienna",
        nonexistent="shift_forward",  # Frühlingssprung
        ambiguous=False,              # Herbst: doppelte Stunde als Winterzeit (CET)
    )
    # 2) ... und TZ wieder entfernen (PyPSA erwartet tz‑naive Snapshots)
    s.index = s.index.tz_localize(None)

    return s

# ------------------------------------------------------------- #
# 2) Smart‑EV‑Algorithmus (index‑sicher)                         #
# ------------------------------------------------------------- #
def window_mask(index, start: time, end: time) -> pd.Series:
    """
    Erzeugt eine boolesche Maske auf dem ORIGINAL-Index.
    Für die Zeitfensterlogik wird temporär Europe/Vienna lokalisiert.
    """
    # temporär tz‑aware Index bauen
    if getattr(index, "tz", None) is None:
        idx_tz = index.tz_localize(
            "Europe/Vienna",
            nonexistent="shift_forward",
            ambiguous=False,
        )
    else:
        idx_tz = index.tz_convert("Europe/Vienna")

    def inside(ts):
        t = ts.time()
        return (start <= t < end) if start < end else (t >= start or t < end)

    vals = [inside(ts) for ts in idx_tz]
    # Wichtig: Maske mit dem ursprünglichen (tz‑naiven) Index zurückgeben
    return pd.Series(vals, index=index)

def smart_ev(base: pd.Series, pv: pd.Series, e_kwh: float, p_kw: float, mask: pd.Series):
    """
    Greedy: lädt zuerst in Zeitpunkte mit größtem PV‑Überschuss.
    Verwendet .isin statt .reindex – dadurch robust bei doppelten Index‑Labels.
    """
    out = pd.Series(0.0, index=base.index)
    h = (out.index[1] - out.index[0]).total_seconds() / 3600.0

    # Index der True‑Mask vorab holen
    true_idx = mask.index[mask]

    for _, grp in out.groupby(out.index.date):
        # bool‑Array für das Tagesfenster – keine Reindexierung nötig
        win_bool = grp.index.isin(true_idx)
        idx = grp.index[win_bool]
        if len(idx) == 0:
            continue

        surplus = (-pv[idx]) - base[idx]
        surplus = surplus.clip(lower=0.0)
        order = surplus.sort_values(ascending=False).index

        rem = e_kwh
        for ts in order:
            if rem <= 1e-3:
                break
            cap = p_kw * h
            ch = min(cap, rem)
            out[ts] = ch / h
            rem -= ch
        if rem > 1e-3:
            for ts in idx:
                if out[ts] > 0:
                    continue
                cap = p_kw * h
                ch = min(cap, rem)
                out[ts] = ch / h
                rem -= ch
                if rem <= 1e-3:
                    break
    return out

# ------------------------------------------------------------- #
# 3) PyPSA‑Netzmodell                                            #
# ------------------------------------------------------------- #
def build_network(p: dict[str, pd.Series], grid_kw: float) -> pypsa.Network:
    """
    Wichtig: Zeitreihen erst NACH dem Hinzufügen der Komponenten setzen.
    So sind Länge und Index garantiert identisch zu n.snapshots.
    """
    n = pypsa.Network()
    snaps = p["load"].index               # tz‑naiv
    n.set_snapshots(snaps)

    n.add("Bus", "grid")
    n.add("Line", "limit", bus0="grid", bus1="grid", s_nom=grid_kw)

    # Komponenten ohne p_set anlegen
    n.add("Load", "demand",   bus="grid")
    n.add("Load", "ev_cars",  bus="grid")
    n.add("Load", "ev_trucks",bus="grid")
    n.add("Generator", "pv",  bus="grid", marginal_cost=0.0)

    # Zeitreihen setzen (sauber ausgerichtet auf snaps)
    n.loads_t.p_set = pd.DataFrame(
        index=snaps,
        data={
            "demand":    p["load"].reindex(snaps).fillna(0.0).values,
            "ev_cars":   p["ev_cars"].reindex(snaps).fillna(0.0).values,
            "ev_trucks": p["ev_trucks"].reindex(snaps).fillna(0.0).values,
        },
    )
    n.generators_t.p_set = pd.DataFrame(
        index=snaps,
        data={
            "pv": -p["pv"].reindex(snaps).fillna(0.0).values
        },
    )

    # Speicher
    n.add(
        "StorageUnit",
        "battery",
        bus="grid",
        p_nom_extendable=True,
        max_hours_extendable=True,
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        capital_cost=0.0,
        marginal_cost=0.0,
    )
    return n

def add_capex(n: pypsa.Network, c_kwh: float, c_kw: float):
    n.objective += c_kwh * n.storage_units["e_nom"].sum() + c_kw * n.storage_units["p_nom"].sum()

# ------------------------------------------------------------- #
# 4) Streamlit‑UI                                                #
# ------------------------------------------------------------- #
st.title("🔋 Optimale Batteriegröße bestimmen")

sb = st.sidebar
sb.header("Basisparameter")
res      = sb.selectbox("Zeitauflösung", ["15min", "60min"], 0)
grid_kw  = sb.number_input("Grid‑Anschluss (kW)", 10, None, 800, 10)
cap_kwh  = sb.number_input("CapEx €/kWh", 0, None, 350)
cap_kw   = sb.number_input("CapEx €/kW", 0, None, 150)

sb.header("CSV‑Uploads")
load_file = sb.file_uploader("Verbrauchs‑CSV (Pflicht)")
pv_files  = sb.file_uploader("PV‑CSV‑Dateien (optional, mehrere)", accept_multiple_files=True)

sb.header("EV‑Ladefenster (Smart)")
smart    = sb.checkbox("Smart‑Charging aktiv", True)
cars_e   = sb.number_input("PKW‑Energie/Tag (kWh)", 0, 2000, 150)
cars_p   = sb.number_input("PKW‑Leistung (kW)", 0, 350, 22)
cars_s   = sb.time_input("PKW‑Start", time(17))
cars_e_t = sb.time_input("PKW‑Ende", time(6))
trucks_e = sb.number_input("LKW‑Energie/Tag (kWh)", 0, 4000, 300)
trucks_p = sb.number_input("LKW‑Leistung (kW)", 0, 1000, 60)
trucks_s = sb.time_input("LKW‑Start", time(20))
trucks_e_t = sb.time_input("LKW‑Ende", time(4))

run = sb.button("🚀 Optimieren")

# ------------------------------------------------------------- #
# 5) Daten einlesen & Optimierung                                #
# ------------------------------------------------------------- #
if run:
    if load_file is None:
        st.error("Bitte Verbrauchs‑CSV hochladen.")
        st.stop()

    try:
        load = read_profile(load_file, res)
    except Exception as e:
        st.error(f"Fehler beim Einlesen der Last‑CSV: {e}")
        st.stop()

    pv = pd.Series(0.0, index=load.index)
    if pv_files:
        for f in pv_files:
            try:
                pv = pv.add(read_profile(f, res), fill_value=0.0)
            except Exception as e:
                st.error(f"Fehler in PV‑Datei {f.name}: {e}")
                st.stop()

    prof = {
        "load": load,
        "pv": pv,
        "ev_cars": pd.Series(0.0, index=load.index),
        "ev_trucks": pd.Series(0.0, index=load.index),
    }

    if smart:
        mask_c = window_mask(load.index, cars_s, cars_e_t)
        net0 = load + pv
        prof["ev_cars"] = smart_ev(net0, pv, cars_e, cars_p, mask_c)
        net1 = net0 + prof["ev_cars"]
        mask_t = window_mask(load.index, trucks_s, trucks_e_t)
        prof["ev_trucks"] = smart_ev(net1, pv, trucks_e, trucks_p, mask_t)

    net = build_network(prof, grid_kw)
    add_capex(net, cap_kwh, cap_kw)

    with st.spinner("Optimierung läuft …"):
        net.optimize()

    e_nom = net.storage_units.loc["battery", "e_nom_opt"]
    p_nom = net.storage_units.loc["battery", "p_nom_opt"]
    c1, c2 = st.columns(2)
    c1.metric("Energie (kWh)", f"{e_nom:.1f}")
    c2.metric("Leistung (kW)", f"{p_nom:.1f}")

    soc = net.storage_units_t["state_of_charge"].loc[:, "battery"]
    grid_imp = (
        net.loads_t["p"].sum(axis=1)
        + net.generators_t["p"].sum(axis=1)
        + soc.diff().fillna(0) / ((soc.index[1] - soc.index[0]).total_seconds() / 3600)
    )

    # Plots
    st.subheader("Zeitreihen")
    tabs = st.tabs(["Load", "PV", "EV Cars", "EV Trucks", "SOC + Grid"])
    plots = [load, -pv, prof["ev_cars"], prof["ev_trucks"]]

    for tab, data in zip(tabs[:-1], plots):
        with tab:
            fig, ax = plt.subplots()
            data.plot(ax=ax)
            ax.set_ylabel("kW")
            st.pyplot(fig)

    with tabs[-1]:
        fig, ax = plt.subplots()
        soc.plot(ax=ax, label="SOC (kWh)")
        ax2 = ax.twinx()
        grid_imp.plot(ax=ax2, label="Grid (kW)")
        ax.set_ylabel("kWh"); ax2.set_ylabel("kW")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        st.pyplot(fig)

    # Download‑CSV
    st.subheader("SOC‑CSV herunterladen")
    buf = io.StringIO()
    soc.to_csv(buf)
    st.download_button("Download CSV", buf.getvalue(), file_name="battery_soc.csv")
