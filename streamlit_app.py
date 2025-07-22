Streamlit‑App für Batterie‑Sizing (Runtime‑safe)
==============================================
*Version 0.2 – ohne HiGHS‑Abhängigkeit; kompatibel mit Python 3.11/3.12*

Interaktives Frontend zur Batterie‑Optimierung mithilfe von PyPSA.
Funktionalität bleibt wie zuvor, jedoch:

* **Keine** explizite `highs`‑/`highspy`‑Abhängigkeit mehr → läuft auf
  Streamlit Community Cloud mit Standard‐GLPK.
* `net.optimize()` überlässt die Solver‑Wahl PyPSA; wenn in der Umgebung
  `highspy` oder `cbc` verfügbar ist, wird dieser genutzt, sonst GLPK.
* Kommentare enthalten minimale **requirements.txt
----------------
```
# Fix: Streamlit 1.35 verlangt numpy<2 → deshalb bleiben wir bei älteren Versionen
streamlit==1.35.0
pandas==1.5.3      # letzte Pandas‑Version, die noch mit numpy<2 funktioniert
numpy==1.24.4
matplotlib==3.8.4
pypsa==0.35.1
```

runtime.txt
-----------
```
python-3.12.3
```
**
  Beispiele zur reibungslosen Bereitstellung.
"""

import io
from datetime import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import streamlit as st

# -----------------------------------------------------------------------------
# Build‑/Deploy‑Hinweis (nicht code‑kritisch)
# -----------------------------------------------------------------------------
# Leg eine Datei `requirements.txt` ins Repo mit u. a.:
#   streamlit==1.35.0
#   pandas==2.3.1
#   numpy==2.3.1
#   matplotlib==3.8.4
#   pypsa==0.35.1
#
# Optional schneller Solver (nur falls Wheel verfügbar):
#   highspy==1.6.2.post10
#
# Zusätzlich `runtime.txt`  für Streamlit Cloud:
#   python-3.12.3
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Hilfsfunktionen (gekürzt, identisch zur vorherigen Version)
# -----------------------------------------------------------------------------

def read_profile(uploaded_file, resolution: str) -> pd.Series:
    df = pd.read_csv(uploaded_file, parse_dates=[0], index_col=0)
    series = (
        df.iloc[:, 0]
        .astype(float)
        .resample(resolution)
        .mean()
        .fillna(0.0)
    )
    series.index = series.index.tz_localize(
        "Europe/Vienna", nonexistent="shift_forward", ambiguous="NaT"
    )
    return series


def make_window_mask(index: pd.DatetimeIndex, start: time, end: time) -> pd.Series:
    def in_window(ts):
        local = ts.tz_convert("Europe/Vienna").time()
        return (start <= local < end) if start < end else (local >= start or local < end)

    return pd.Series([in_window(ts) for ts in index], index=index)


def smart_ev_profile(base_net: pd.Series, pv: pd.Series, energy_kwh: float, power_kw: float, mask: pd.Series) -> pd.Series:
    profile = pd.Series(0.0, index=base_net.index)
    step_h = (profile.index[1] - profile.index[0]).total_seconds() / 3600.0

    for _, day_idx in profile.groupby(profile.index.date):
        win = day_idx.index[mask[day_idx.index]]
        if win.empty:
            continue
        surplus = (-pv[win]) - base_net[win]
        surplus.clip(lower=0.0, inplace=True)
        order = surplus.sort_values(ascending=False).index
        remaining = energy_kwh
        for ts in order:
            if remaining <= 1e-3:
                break
            cap = power_kw * step_h
            charge = min(cap, remaining)
            profile[ts] = charge / step_h
            remaining -= charge
        if remaining > 1e-3:
            for ts in win:
                if profile[ts] > 0:
                    continue
                cap = power_kw * step_h
                charge = min(cap, remaining)
                profile[ts] = charge / step_h
                remaining -= charge
                if remaining <= 1e-3:
                    break
    return profile


def build_network(profiles, grid_kw):
    n = pypsa.Network()
    n.set_snapshots(profiles["load"].index)
    n.add("Bus", "grid")
    n.add("Line", "grid_limit", bus0="grid", bus1="grid", s_nom=grid_kw)
    n.add("Load", "demand", bus="grid", p_set=profiles["load"].values)
    n.add("Generator", "pv", bus="grid", p_set=-profiles["pv"].values, marginal_cost=0.0)
    n.add("Load", "ev_cars", bus="grid", p_set=profiles["ev_cars"].values)
    n.add("Load", "ev_trucks", bus="grid", p_set=profiles["ev_trucks"].values)
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


def add_battery_costs(n, cost_kwh, cost_kw):
    n.objective += cost_kwh * n.storage_units["e_nom"].sum() + cost_kw * n.storage_units["p_nom"].sum()

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Battery Sizing Tool", layout="wide")
st.title("🔋 Optimale Batteriegröße bestimmen")

st.sidebar.header("Basisparameter")
resolution = st.sidebar.selectbox("Zeitauflösung", ["15min", "60min"], index=0)
gridsize = st.sidebar.number_input("Grid‑Anschluss (kW)", min_value=10, value=800, step=10)
capex_kwh = st.sidebar.number_input("CapEx €/kWh", 0, None, 350)
capex_kw = st.sidebar.number_input("CapEx €/kW", 0, None, 150)

st.sidebar.header("CSV‑Uploads")
load_file = st.sidebar.file_uploader("Verbrauchs‑CSV")
pv_file = st.sidebar.file_uploader("PV‑CSV (optional)")

st.sidebar.header("EV‑Ladefenster (Smart)")
smart_ev = st.sidebar.checkbox("Smart‑Charging aktiv", True)

cars_energy = st.sidebar.number_input("PKW‑Energie/Tag (kWh)", 0, 2000, 150)
cars_power = st.sidebar.number_input("PKW‑Ladeleistung (kW)", 0, 350, 22)
cars_start = st.sidebar.time_input("PKW‑Start", time(17, 0))
cars_end = st.sidebar.time_input("PKW‑Ende", time(6, 0))

trucks_energy = st.sidebar.number_input("LKW‑Energie/Tag (kWh)", 0, 4000, 300)
trucks_power = st.sidebar.number_input("LKW‑Ladeleistung (kW)", 0, 1000, 60)
trucks_start = st.sidebar.time_input("LKW‑Start", time(20, 0))
trucks_end = st.sidebar.time_input("LKW‑Ende", time(4, 0))

run_btn = st.sidebar.button("🚀 Optimieren")

if run_btn:
    if load_file is None:
        st.error("Bitte Verbrauchs‑Profil hochladen.")
        st.stop()

    load = read_profile(load_file, resolution)
    pv = read_profile(pv_file, resolution) if pv_file else pd.Series(0.0, index=load.index)

    profiles = {"load": load, "pv": pv, "ev_cars": pd.Series(0.0, index=load.index), "ev_trucks": pd.Series(0.0, index=load.index)}

    if smart_ev:
        mask_cars = make_window_mask(load.index, cars_start, cars_end)
        net_before = load + pv
        profiles["ev_cars"] = smart_ev_profile(net_before, pv, cars_energy, cars_power, mask_cars)
        net_before += profiles["ev_cars"]
        mask_trucks = make_window_mask(load.index, trucks_start, trucks_end)
        profiles["ev_trucks"] = smart_ev_profile(net_before, pv, trucks_energy, trucks_power, mask_trucks)

    net = build_network(profiles, gridsize)
    add_battery_costs(net, capex_kwh, capex_kw)

    with st.spinner("Running PyPSA optimisation…"):
        net.optimize()  # kein expliziter Solver

    e_nom = net.storage_units.loc["battery", "e_nom_opt"]
    p_nom = net.storage_units.loc["battery", "p_nom_opt"]

    c1, c2 = st.columns(2)
    c1.metric("Batterie‑Energie (kWh)", f"{e_nom:.1f}")
    c2.metric("Batterie‑Leistung (kW)", f"{p_nom:.1f}")

    soc = net.storage_units_t["state_of_charge"].loc[:, "battery"]
    grid_import = net.loads_t["p"].sum(axis=1) + net.generators_t["p"].sum(axis=1) + soc.diff().fillna(0) / ((soc.index[1] - soc.index[0]).total_seconds() / 3600.0)

    st.subheader("Zeitreihen‑Plots")
    tabs = st.tabs(["Load", "PV", "EV‑Cars", "EV‑Trucks", "SOC & Grid"])
    datasets = [load, -pv, profiles["ev_cars"], profiles["ev_trucks"], None]

    for tab, data in zip(tabs[:-1], datasets[:-1]):
        with tab:
            fig, ax = plt.subplots()
            data.plot(ax=ax)
            ax.set_ylabel("kW")
            st.pyplot(fig)

    with tabs[-1]:
        fig, ax = plt.subplots()
        soc.plot(ax=ax, label="SOC (kWh)")
        ax2 = ax.twinx()
        grid_import.plot(ax=ax2, color="orange", label="Grid (kW)")
        ax.set_ylabel("kWh"); ax2.set_ylabel("kW")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        st.pyplot(fig)

    st.subheader("Download SOC‑CSV")
    buf = io.StringIO(); soc.to_csv(buf)
    st.download_button("CSV herunterladen", buf.getvalue(), file_name="battery_soc.csv")
