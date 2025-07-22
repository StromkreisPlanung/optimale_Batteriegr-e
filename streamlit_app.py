"""
Streamlit‑App für Batterie‑Sizing (runtime‑safe)
================================================
* keine HiGHS‑Pflicht – PyPSA wählt GLPK oder vorhandenen Solver
* läuft auf Streamlit Community Cloud mit Python 3.12
"""

from __future__ import annotations

import io
from datetime import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import streamlit as st


# ------------------------------------------------------------------ #
# Helper functions                                                   #
# ------------------------------------------------------------------ #
def read_profile(upload, res: str) -> pd.Series:
    df = pd.read_csv(upload, parse_dates=[0], index_col=0)
    s = df.iloc[:, 0].astype(float).resample(res).mean().fillna(0.0)
    s.index = s.index.tz_localize("Europe/Vienna", nonexistent="shift_forward", ambiguous="NaT")
    return s


def window_mask(index, start: time, end: time) -> pd.Series:
    def inside(ts):
        t = ts.tz_convert("Europe/Vienna").time()
        return (start <= t < end) if start < end else (t >= start or t < end)

    return pd.Series([inside(ts) for ts in index], index=index)


def smart_ev(base: pd.Series, pv: pd.Series, en_kwh: float, p_kw: float, mask: pd.Series):
    out = pd.Series(0.0, index=base.index)
    h = (out.index[1] - out.index[0]).total_seconds() / 3600.0
    for _, grp in out.groupby(out.index.date):
        idx = grp.index[mask[grp.index]]
        if idx.empty:
            continue
        surplus = (-pv[idx]) - base[idx]
        surplus.clip(lower=0.0, inplace=True)
        order = surplus.sort_values(ascending=False).index
        rem = en_kwh
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


def network(p: dict[str, pd.Series], grid_kw):
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


def capex(n, c_kwh, c_kw):
    n.objective += c_kwh * n.storage_units["e_nom"].sum() + c_kw * n.storage_units["p_nom"].sum()


# ------------------------------------------------------------------ #
# Streamlit UI                                                      #
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Battery Sizing Tool", layout="wide")
st.title("🔋 Optimale Batteriegröße bestimmen")

sidebar = st.sidebar
sidebar.header("Basisparameter")
res = sidebar.selectbox("Zeitauflösung", ["15min", "60min"], 0)
grid_kw = sidebar.number_input("Grid‑Anschluss (kW)", 10, None, 800, 10)
cap_kwh = sidebar.number_input("CapEx €/kWh", 0, None, 350)
cap_kw = sidebar.number_input("CapEx €/kW", 0, None, 150)

sidebar.header("CSV‑Uploads")
load_file = sidebar.file_uploader("Verbrauch CSV")
pv_file = sidebar.file_uploader("PV CSV (optional)")

sidebar.header("EV‑Ladefenster (Smart)")
smart = sidebar.checkbox("Smart‑Charging aktiv", True)
cars_e = sidebar.number_input("PKW‑Energie/Tag (kWh)", 0, 2000, 150)
cars_p = sidebar.number_input("PKW‑Leistung (kW)", 0, 350, 22)
cars_s = sidebar.time_input("PKW‑Start", time(17))
cars_e_t = sidebar.time_input("PKW‑Ende", time(6))
trucks_e = sidebar.number_input("LKW‑Energie/Tag (kWh)", 0, 4000, 300)
trucks_p = sidebar.number_input("LKW‑Leistung (kW)", 0, 1000, 60)
trucks_s = sidebar.time_input("LKW‑Start", time(20))
trucks_e_t = sidebar.time_input("LKW‑Ende", time(4))

run = sidebar.button("🚀 Optimieren")

if run:
    if load_file is None:
        st.error("Bitte Verbrauchs‑CSV hochladen.")
        st.stop()

    load = read_profile(load_file, res)
    pv = read_profile(pv_file, res) if pv_file else pd.Series(0.0, index=load.index)

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

    net = network(prof, grid_kw)
    capex(net, cap_kwh, cap_kw)

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

    st.subheader("SOC‑CSV herunterladen")
    buf = io.StringIO()
    soc.to_csv(buf)
    st.download_button("Download CSV", buf.getvalue(), file_name="battery_soc.csv")
