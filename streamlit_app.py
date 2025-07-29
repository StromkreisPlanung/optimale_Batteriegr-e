from __future__ import annotations
"""
streamlit_app.py · Battery‑Sizing Dashboard  v0.6.9
===================================================
• Mehrere PV‑CSV‑Dateien werden addiert
• Robuster CSV‑Reader (Semikolon/Komma, Dezimalpunkt/-komma)
• DST-Fix (Europe/Vienna), PyPSA bekommt tz‑naive Snapshots
• Smart‑EV ohne reindex (duplikat‑sicher)
• Page‑Config‑Guard
• Matplotlib-Fallback + Versionsanzeige
• CapEx: direkt an StorageUnit über capital_cost = cap_kw + cap_kwh * max_hours
  → Optimiert kW; Energie = p_nom_opt * max_hours
• SOC-Ermittlung robust (Series/DataFrame) → kein KeyError auf "battery"
"""

import io
from datetime import time
import importlib.util
import sys

import pandas as pd
import pypsa
import streamlit as st

# --- Matplotlib optional laden (Fallback auf Streamlit-Charts) ---
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

# --- Page Config (einmalig) ---
try:
    st.set_page_config(page_title="Battery Sizing Tool", layout="wide")
except st.errors.StreamlitAPIException:
    pass

# --- Sidebar: Diagnose der Umgebung ---
def _mod_ver(name):
    spec = importlib.util.find_spec(name)
    if not spec:
        return "not installed"
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "installed (no __version__)"

with st.sidebar.expander("⚙️ Umgebung", expanded=False):
    st.write(
        f"Python: {sys.version.split()[0]}\n"
        f"streamlit: {st.__version__}\n"
        f"pandas: {_mod_ver('pandas')}\n"
        f"numpy:  {_mod_ver('numpy')}\n"
        f"matplotlib: {_mod_ver('matplotlib')}\n"
        f"pypsa: {_mod_ver('pypsa')}"
    )

# ---------------------- CSV-Reader ---------------------- #
def read_profile(upload, res: str) -> pd.Series:
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
        dayfirst=semicolon,
        engine="python",
    )
    # Doppelte Timestamps mitteln
    df = df.groupby("datetime", as_index=False)["power_kw"].mean()
    df.set_index("datetime", inplace=True)

    s = df["power_kw"].resample(res).mean().fillna(0.0)

    # TZ anwenden (DST), dann wieder tz-naiv für PyPSA
    s.index = s.index.tz_localize(
        "Europe/Vienna",
        nonexistent="shift_forward",
        ambiguous=False,
    )
    s.index = s.index.tz_localize(None)
    return s

# ------------------ Smart-EV (index-sicher) -------------- #
def window_mask(index, start: time, end: time) -> pd.Series:
    # temporär tz-aware, um Tagesfenster korrekt zu prüfen
    if getattr(index, "tz", None) is None:
        idx_tz = index.tz_localize("Europe/Vienna", nonexistent="shift_forward", ambiguous=False)
    else:
        idx_tz = index.tz_convert("Europe/Vienna")

    def inside(ts):
        t = ts.time()
        return (start <= t < end) if start < end else (t >= start or t < end)

    vals = [inside(ts) for ts in idx_tz]
    return pd.Series(vals, index=index)  # Maske bleibt auf Originalindex

def smart_ev(base: pd.Series, pv: pd.Series, e_kwh: float, p_kw: float, mask: pd.Series):
    out = pd.Series(0.0, index=base.index)
    h = (out.index[1] - out.index[0]).total_seconds() / 3600.0
    true_idx = mask.index[mask]

    for _, grp in out.groupby(out.index.date):
        idx = grp.index[grp.index.isin(true_idx)]
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

# ------------------- PyPSA-Netz (defensiv) ---------------- #
def build_network(p: dict[str, pd.Series], grid_kw: float, cap_kwh: float, cap_kw: float, h_batt: float) -> pypsa.Network:
    """Zeitreihen defensiv bereinigen und erst NACH dem Add setzen.
       CapEx direkt an StorageUnit über capital_cost = cap_kw + cap_kwh * h_batt.
    """
    def sanitize(s: pd.Series, snaps: pd.DatetimeIndex) -> pd.Series:
        s = s.groupby(level=0).mean()
        s = s.sort_index()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s.reindex(snaps).fillna(0.0)

    n = pypsa.Network()
    snaps = pd.Index(p["load"].index).drop_duplicates(keep="first").sort_values()
    n.set_snapshots(snaps)

    n.add("Bus", "grid")
    n.add("Line", "limit", bus0="grid", bus1="grid", s_nom=grid_kw)
    n.add("Load", "demand", bus="grid")
    n.add("Load", "ev_cars", bus="grid")
    n.add("Load", "ev_trucks", bus="grid")
    n.add("Generator", "pv", bus="grid", marginal_cost=0.0)

    load_s   = sanitize(p["load"], snaps)
    ev_cars  = sanitize(p["ev_cars"], snaps)
    ev_truck = sanitize(p["ev_trucks"], snaps)
    pv_s     = sanitize(p["pv"], snaps)

    n.loads_t.p_set = pd.DataFrame(
        index=snaps,
        data={
            "demand":    load_s.values,
            "ev_cars":   ev_cars.values,
            "ev_trucks": ev_truck.values,
        },
    )
    n.generators_t.p_set = pd.DataFrame(index=snaps, data={"pv": (-pv_s).values})

    # Batterie: feste Dauer (max_hours) und effektiver €/kW-Kostensatz
    cap_eff_kw = cap_kw + cap_kwh * h_batt  # €/kW_eff
    n.add(
        "StorageUnit",
        "battery",
        bus="grid",
        p_nom_extendable=True,
        max_hours=h_batt,                # feste Energiedauer
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        capital_cost=cap_eff_kw,         # Kosten wirken auf p_nom_opt
        marginal_cost=0.0,
    )
    return n

# ----------------------- UI ------------------------------- #
st.title("🔋 Optimale Batteriegröße bestimmen")

sb = st.sidebar
sb.header("Basisparameter")
res      = sb.selectbox("Zeitauflösung", ["15min", "60min"], 0)
grid_kw  = sb.number_input("Grid‑Anschluss (kW)", min_value=10, value=800, step=10)
cap_kwh  = sb.number_input("CapEx €/kWh", min_value=0, value=350)
cap_kw   = sb.number_input("CapEx €/kW",  min_value=0, value=150)
h_batt   = sb.number_input("Batteriedauer max_hours (h)", min_value=0.25, max_value=12.0, value=2.0, step=0.25)

sb.header("CSV‑Uploads")
load_file = sb.file_uploader("Verbrauchs‑CSV (Pflicht)")
pv_files  = sb.file_uploader("PV‑CSV‑Dateien (optional, mehrere)", accept_multiple_files=True)

sb.header("EV‑Ladefenster (Smart)")
smart    = sb.checkbox("Smart‑Charging aktiv", True)
cars_e   = sb.number_input("PKW‑Energie/Tag (kWh)", min_value=0, max_value=2000, value=150)
cars_p   = sb.number_input("PKW‑Leistung (kW)",     min_value=0, max_value=350,  value=22)
cars_s   = sb.time_input("PKW‑Start", time(17))
cars_e_t = sb.time_input("PKW‑Ende",  time(6))
trucks_e = sb.number_input("LKW‑Energie/Tag (kWh)", min_value=0, max_value=4000, value=300)
trucks_p = sb.number_input("LKW‑Leistung (kW)",     min_value=0, max_value=1000, value=60)
trucks_s = sb.time_input("LKW‑Start", time(20))
trucks_e_t = sb.time_input("LKW‑Ende", time(4))

run = sb.button("🚀 Optimieren")

# ---------------- Daten einlesen & Optimierung ------------- #
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

    net = build_network(prof, grid_kw, cap_kwh, cap_kw, h_batt)

    with st.spinner("Optimierung läuft …"):
        net.optimize()

    # Ergebnisse: p_nom_opt direkt aus Netz; e_nom = p_nom_opt * h_batt
    p_nom = net.storage_units.loc["battery", "p_nom_opt"]
    e_nom = p_nom * h_batt

    c1, c2 = st.columns(2)
    c1.metric("Energie (kWh)", f"{e_nom:.1f}")
    c2.metric("Leistung (kW)", f"{p_nom:.1f}")

    # --- SOC robust ermitteln (Series oder DataFrame) ---
    soc_tbl = net.storage_units_t.get("state_of_charge")
    if soc_tbl is None or len(soc_tbl) == 0:
        st.error("Kein SOC in den Ergebnissen – Optimierung evtl. fehlgeschlagen.")
        st.stop()

    if isinstance(soc_tbl, pd.Series):
        soc = soc_tbl.rename("battery")
    else:
        if "battery" in soc_tbl.columns:
            soc = soc_tbl["battery"]
        else:
            first_col = soc_tbl.columns[0]
            soc = soc_tbl[first_col].rename(first_col)

    # Approx. Grid-Import: Lasten + Generatoren + d(SOC)/dt
    dt_h = (soc.index[1] - soc.index[0]).total_seconds() / 3600
    grid_imp = (
        net.loads_t["p"].sum(axis=1)
        + net.generators_t["p"].sum(axis=1)
        + soc.diff().fillna(0) / dt_h
    )

    st.subheader("Zeitreihen")
    tabs = st.tabs(["Load", "PV", "EV Cars", "EV Trucks", "SOC + Grid"])
    plots = [load, -pv, prof["ev_cars"], prof["ev_trucks"]]

    for tab, data in zip(tabs[:-1], plots):
        with tab:
            if plt is None:
                st.line_chart(data.rename("kW"))
            else:
                fig, ax = plt.subplots()
                data.plot(ax=ax)
                ax.set_ylabel("kW")
                st.pyplot(fig)

    with tabs[-1]:
        if plt is None:
            st.line_chart(soc.rename("SOC (kWh)"))
            st.line_chart(grid_imp.rename("Grid (kW)"))
        else:
            fig, ax = plt.subplots()
            soc.plot(ax=ax, label="SOC (kWh)")
            ax2 = ax.twinx()
            grid_imp.plot(ax=ax2, label="Grid (kW)")
            ax.set_ylabel("kWh"); ax2.set_ylabel("kW")
            ax.legend(loc="upper left"); ax2.legend(loc="upper right")
            st.pyplot(fig)

    st.subheader("SOC‑CSV herunterladen")
    buf = io.StringIO()
    soc.to_csv(buf)
    st.download_button("Download CSV", buf.getvalue(), file_name="battery_soc.csv")
