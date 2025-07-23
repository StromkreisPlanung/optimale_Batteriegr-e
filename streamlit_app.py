from __future__ import annotations
"""
streamlit_app.py – Battery‑Sizing Dashboard  v0.5
=================================================
• akzeptiert beliebig viele PV‑CSV‑Dateien (werden aufsummiert)
• CSV‑Auto‑Erkennung (; + ,  oder , + .)
• Smart‑EV‑Charging mit Index‑Align‑Fix
• korrekte Klammern in grid_imp‑Berechnung
"""

import io
from datetime import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import streamlit as st

# ------------------------------------------------------------------
# 1. CSV‑Reader
# ------------------------------------------------------------------
def read_profile(upload, res: str) -> pd.Series:
    raw = upload.getvalue().decode("utf-8-sig")
    header = raw.splitlines()[0]
    semicolon = ";" in header
    sep = ";" if semicolon else ","
    dec = "," if semicolon else "."

    df = pd.read_csv(
        io.StringIO(raw),
        sep=sep,
        decimal=dec,
        names=["datetime", "power_kw"],
        header=0,
        parse_dates=["datetime"],
        dayfirst=semicolon,
        engine="python",
    )
    df.set_index("datetime", inplace=True)
    s = df["power_kw"].resample(res).mean().fillna(0.0)
    s.index = s.index.tz_localize(
        "Europe/Vienna", nonexistent="shift_forward", ambiguous="NaT"
    )
    return s


# ------------------------------------------------------------------
# 2. Smart‑EV‑Algorithmus
# ------------------------------------------------------------------
def window_mask(index, start: time, end: time) -> pd.Series:
    def inside(ts):
        t = ts.tz_convert("Europe/Vienna").time()
        return (start <= t < end) if start < end else (t >= start or t < end)

    return pd.Series([inside(ts) for ts in index], index=index)


def smart_ev(base: pd.Series, pv: pd.Series, e_kwh: float, p_kw: float, mask: pd.Series):
    out = pd.Series(0.0, index=base.index)
    h = (out.index[1] - out.index[0]).total_seconds() / 3600

    for _, grp in out.groupby(out.index.date):
        win = mask.reindex(grp.index, fill_value=False)  # Index‑Align‑Fix
        idx = grp.index[win]
        if idx.empty:
            continue
        surplus = (-pv[idx]) - base[idx]
        surplus.clip(lower=0.0, inplace=True)
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


# ------------------------------------------------------------------
# 3. PyPSA‑Netzmodell
# ------------------------------------------------------------------
def build_network(p: dict[str, pd.Series], grid_kw: float) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(p["load"].index)
    n.add("Bus", "grid")
    n.add("Line", "limit", bus0="grid", bus1="grid", s_nom=grid_kw)
    n.add("Load", "demand", bus="grid", p_set=p["load"].values)
    n.add("Generator", "pv", bus="grid", p_set=-p["pv"].values, marginal_cost=0.0)
    n.add("Load", "ev_cars", bus="grid", p_set=p["ev_cars"].values)
    n.add("Load", "ev_trucks", bus="grid", p_set=p["ev_trucks"].values)
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


# ------------------------------------------------------------------
# 4. Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Battery Sizing Tool", layout="wide")
st.title("🔋 Optimale Batteriegröße bestimmen")

sb = st.sidebar
sb.header("Basisparameter")
res = sb.selectbox("Zeitauflösung", ["15min", "60min"], 0)
grid_kw = sb.number_input("Grid‑Anschluss (kW)", 10, None, 800, 10)
cap_kwh = sb.number_input("CapEx €/kWh", 0, None, 350)
cap_kw = sb.number_input("CapEx €/kW", 0, None, 150)

sb.header("CSV‑Uploads")
load_file = sb.file_uploader("Verbrauchs‑CSV (Pflicht)")
pv_files = sb.file_uploader(
    "PV‑CSV‑Dateien (optional, mehrere)", accept_multiple_files=True
)

sb.header("EV‑Ladefenster (Smart)")
smart = sb.checkbox("Smart‑Charging aktiv", True)

cars_e = sb.number_input("PKW‑Energie/Tag (kWh)", 0, 2000, 150)
cars_p = sb.number_input("PKW‑Leistung (kW)", 0, 350, 22)
cars_s = sb.time_input("PKW‑Start", time(17))
cars_e_t = sb.time_input("PKW‑Ende", time(6))

trucks_e = sb.number_input("LKW‑Energie/Tag (kWh)", 0, 4000, 300)
trucks_p = sb.number_input("LKW‑Leistung (kW)", 0, 1000, 60)
trucks_s = sb.time_input("LKW‑Start", time(20))
trucks_e_t = sb.time_input("LKW‑Ende", time(4))

run = sb.button("🚀 Optimieren")

# ------------------------------------------------------------------
# 5. Daten einlesen & Optimierung
# ------------------------------------------------------------------
if run:
    if load_file is None:
        st.error("Bitte Verbrauchs‑CSV hochladen.")
        st.stop()

    # Load
    try:
        load = read_profile(load_file, res)
    except Exception as e:
        st.error(f"Fehler beim Einlesen der Last‑CSV: {e}")
        st.stop()

    # PV (Summe mehrerer Dateien)
    pv = pd.Series(0.0, index=load.index)
    if pv_files:
        for f in pv_files:
            try:
                pv = pv.add(read_profile(f, res), fill_value=0.0)
            except Exception as e:
                st.error(f"Fehler in PV‑Datei {f.name}: {e}")
                st.stop()

    # Profile‑Dict
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

    # PyPSA‑Optimierung
    net = build_network(prof, grid_kw)
    add_capex(net, cap_kwh, cap_kw)

    with st.spinner("Optimierung läuft …"):
        net.optimize()

    # Ergebnisse
    e_nom = net.storage_units.loc["battery", "e_nom_opt"]
    p_nom = net.storage_units.loc["battery", "p_nom_opt"]
    c1, c2 = st.columns(2)
    c1.metric("Energie (kWh)", f"{e_nom:.1f}")
    c2.metric("Leistung (kW)", f"{p_nom:.1f}")

    soc = net.storage_units_t["state_of_charge"].loc[:, "battery"]
    grid_imp = (
        net.loads_t["p"].sum(axis=1)
        + net.generators_t["p"].sum(axis=1)
        + soc.diff().fillna(0)
          / ((soc.index[1] - soc.index[0]).total_seconds() / 3600)
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
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
        grid_imp.plot(ax=ax2, color="orange", label="Grid (kW)")
        ax.set_ylabel("kWh")
        ax2.set_ylabel("kW")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        st.pyplot(fig)

    # Download‑CSV
    st.subheader("SOC‑CSV herunterladen")
    buf = io.StringIO()
    soc.to_csv(buf)
    st.download_button("Download CSV", buf.getvalue(), file_name="battery_soc.csv")
