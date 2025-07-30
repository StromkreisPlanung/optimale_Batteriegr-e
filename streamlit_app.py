from __future__ import annotations
"""
streamlit_app.py · Battery‑Sizing Dashboard  v0.8.0
===================================================
• Mehrere PV‑CSV (aufsummiert)
• Robuster CSV‑Reader (Semikolon/Komma, Dezimalpunkt/-komma) + DST Fix
• PyPSA mit tz‑naiven Snapshots + korrekter Zeitgewichtung (h)
• Smart‑EV lädt bei PV‑Überschuss (residuale Last < 0)
• PV als positive Erzeugung (Signatur korrigiert)
• CapEx: capital_cost = cap_kw + cap_kwh * max_hours (€/kW_eff)
• NEU: Strompreis (€/kWh) – dispatchbarer Netz‑Import mit marginal_cost
• Kennzahlen: p_nom, e_nom, Netzbezug (kWh) & Energiekosten (EUR)
"""

import io
from datetime import time
import importlib.util
import sys

import pandas as pd
import pypsa
import streamlit as st

# --- Matplotlib optional (Fallback: Streamlit Charts) ---
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

# --- Page Config (einmalig) ---
try:
    st.set_page_config(page_title="Battery Sizing Tool", layout="wide")
except st.errors.StreamlitAPIException:
    pass

# --- Sidebar: Umgebungsinfo ---
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
        dayfirst=semicolon,   # deutsch: dd.mm.yyyy
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

# ------------------ Smart‑EV (residuale Logik) ---------- #
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
    return pd.Series(vals, index=index)

def smart_ev(load_kw: pd.Series, pv_kw: pd.Series, e_kwh: float, p_kw: float, mask: pd.Series):
    """
    Verteilt e_kwh pro Tag im Fenster 'mask', priorisiert Zeiten mit PV‑Überschuss
    (residual = load - pv < 0). Greedy, dann Rest auffüllen.
    """
    out = pd.Series(0.0, index=load_kw.index)
    h = (out.index[1] - out.index[0]).total_seconds() / 3600.0
    true_idx = mask.index[mask]

    for _, grp in out.groupby(out.index.date):
        idx = grp.index[grp.index.isin(true_idx)]
        if len(idx) == 0:
            continue
        residual = (load_kw[idx] - pv_kw[idx])
        surplus = (-residual).clip(lower=0.0)  # >0 bei PV‑Überschuss
        order = surplus.sort_values(ascending=False).index

        rem = e_kwh
        # 1) Slots mit größtem Überschuss
        for ts in order:
            if rem <= 1e-3:
                break
            cap = p_kw * h
            ch = min(cap, rem)
            out[ts] = ch / h
            rem -= ch
        # 2) Rest gleichmäßig im Fenster
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

# ------------------- PyPSA‑Netz (mit Preis) -------------- #
def build_network(p: dict[str, pd.Series],
                  grid_kw: float,
                  cap_kwh: float,
                  cap_kw: float,
                  h_batt: float,
                  price_buy_eur_per_kwh: float) -> pypsa.Network:
    """Zeitreihen defensiv, Snapshots + Zeitgewichtung (h), Netz‑Import mit kWh‑Preis."""
    def sanitize(s: pd.Series, snaps: pd.DatetimeIndex) -> pd.Series:
        s = s.groupby(level=0).mean()
        s = s.sort_index()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s.reindex(snaps).fillna(0.0)

    n = pypsa.Network()
    snaps = pd.Index(p["load"].index).drop_duplicates(keep="first").sort_values()
    n.set_snapshots(snaps)

    # Zeitgewichtung: Dauer je Snapshot (h)
    if len(snaps) >= 2:
        dt_h = (snaps[1] - snaps[0]).total_seconds() / 3600.0
    else:
        dt_h = 1.0
    # In PyPSA 0.35 reicht eine Series:
    n.snapshot_weightings = pd.Series(dt_h, index=snaps)

    n.add("Bus", "grid")

    # Lasten
    n.add("Load", "demand", bus="grid")
    n.add("Load", "ev_cars", bus="grid")
    n.add("Load", "ev_trucks", bus="grid")

    # PV als positive Erzeugung (kostenlos, fix)
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
    n.generators_t.p_set = pd.DataFrame(index=snaps, data={"pv": pv_s.values})

    # Netz‑Import als dispatchbarer Generator mit €/kWh, Kapazität = grid_kw
    n.add(
        "Generator",
        "grid_import",
        bus="grid",
        p_nom=grid_kw,            # Import‑Leistungsgrenze
        p_nom_extendable=False,
        marginal_cost=price_buy_eur_per_kwh,  # €/kWh (da wir kW & h nutzen)
        p_max_pu=1.0,
        p_min_pu=0.0,
    )

    # Batterie: feste Dauer (max_hours) & effektiver €/kW‑Kostensatz
    cap_eff_kw = cap_kw + cap_kwh * h_batt  # €/kW_eff
    n.add(
        "StorageUnit",
        "battery",
        bus="grid",
        p_nom_extendable=True,
        max_hours=h_batt,
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        capital_cost=cap_eff_kw,
        marginal_cost=0.0,
    )
    return n

# ----------------------- UI ------------------------------- #
st.title("🔋 Optimale Batteriegröße bestimmen")

sb = st.sidebar
sb.header("Basisparameter")
res       = sb.selectbox("Zeitauflösung", ["15min", "60min"], 0)
grid_kw   = sb.number_input("Grid‑Anschluss (kW)", min_value=10, value=800, step=10)
cap_kwh   = sb.number_input("CapEx €/kWh", min_value=0.0, value=300.0, step=10.0)
cap_kw    = sb.number_input("CapEx €/kW",  min_value=0.0, value=150.0, step=10.0)
h_batt    = sb.number_input("Batteriedauer max_hours (h)", min_value=0.25, max_value=12.0, value=2.0, step=0.25)
price_buy = sb.number_input("Strompreis Netz (€/kWh)", min_value=0.0, value=0.25, step=0.01, format="%.2f")

sb.header("CSV‑Uploads")
load_file = sb.file_uploader("Verbrauchs‑CSV (Pflicht)")
pv_files  = sb.file_uploader("PV‑CSV‑Dateien (optional, mehrere)", accept_multiple_files=True)

sb.header("EV‑Ladefenster (Smart)")
smart      = sb.checkbox("Smart‑Charging aktiv", True)
cars_e     = sb.number_input("PKW‑Energie/Tag (kWh)", min_value=0, max_value=2000, value=150)
cars_p     = sb.number_input("PKW‑Leistung (kW)",     min_value=0, max_value=350,  value=22)
cars_s     = sb.time_input("PKW‑Start", time(17))
cars_e_t   = sb.time_input("PKW‑Ende",  time(6))
trucks_e   = sb.number_input("LKW‑Energie/Tag (kWh)", min_value=0, max_value=4000, value=300)
trucks_p   = sb.number_input("LKW‑Leistung (kW)",     min_value=0, max_value=1000, value=60)
trucks_s   = sb.time_input("LKW‑Start", time(20))
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
        "pv": pv,  # positiv (Erzeugung)
        "ev_cars": pd.Series(0.0, index=load.index),
        "ev_trucks": pd.Series(0.0, index=load.index),
    }

    if smart:
        mask_c = window_mask(load.index, cars_s, cars_e_t)
        prof["ev_cars"] = smart_ev(load, pv, cars_e, cars_p, mask_c)
        net_after_cars = load + prof["ev_cars"]
        mask_t = window_mask(load.index, trucks_s, trucks_e_t)
        prof["ev_trucks"] = smart_ev(net_after_cars, pv, trucks_e, trucks_p, mask_t)

    # Netz bauen & optimieren
    net = build_network(prof, grid_kw, cap_kwh, cap_kw, h_batt, price_buy)

    with st.spinner("Optimierung läuft …"):
        net.optimize()

    # Ergebnisse
    p_nom = float(net.storage_units.loc["battery", "p_nom_opt"])
    e_nom = p_nom * h_batt
    c1, c2, c3 = st.columns(3)
    c1.metric("Energie (kWh)", f"{e_nom:.1f}")
    c2.metric("Leistung (kW)", f"{p_nom:.1f}")

    # Netz‑Import‑Energie und Kosten
    gi = net.generators_t["p"].get("grid_import")
    if gi is not None is not ...:
        # kW * h = kWh (Zeitgewichtung ist bereits gesetzt)
        if len(net.snapshots) >= 2:
            dt_h = (net.snapshots[1] - net.snapshots[0]).total_seconds() / 3600.0
        else:
            dt_h = 1.0
        energy_kwh = float((gi * dt_h).sum())
        cost_eur   = energy_kwh * price_buy
        c3.metric("Netzbezug / Kosten", f"{energy_kwh:,.0f} kWh / {cost_eur:,.0f} €")
    else:
        c3.metric("Netzbezug / Kosten", "–")

    # --- SOC robust ermitteln (Series/DF/leer) ---
    soc_tbl = net.storage_units_t.get("state_of_charge")
    if soc_tbl is None or len(soc_tbl) == 0:
        soc = pd.Series(0.0, index=net.snapshots, name="battery")
    elif isinstance(soc_tbl, pd.Series):
        soc = soc_tbl.rename("battery")
    else:
        cols = list(getattr(soc_tbl, "columns", []))
        if "battery" in cols:
            soc = soc_tbl["battery"]
        elif len(cols) >= 1:
            soc = soc_tbl.iloc[:, 0].rename(str(cols[0]))
        else:
            soc = pd.Series(0.0, index=net.snapshots, name="battery")

    # Residuale Last und Netz‑Import für Plot
    residual = (prof["load"] + prof["ev_cars"] + prof["ev_trucks"] - prof["pv"]).reindex(net.snapshots).fillna(0.0)
    grid_imp_series = net.generators_t["p"].get("grid_import")
    if grid_imp_series is None:
        # Fallback‑Approximation
        if len(soc.index) >= 2:
            dt_h = (soc.index[1] - soc.index[0]).total_seconds() / 3600
        else:
            dt_h = 1.0
        grid_imp_series = residual + soc.diff().fillna(0) / dt_h

    # ---- Plots ----
    st.subheader("Zeitreihen")
    tabs = st.tabs(["Load", "PV", "EV Cars", "EV Trucks", "SOC + Grid", "Residual"])
    plots = [prof["load"], prof["pv"], prof["ev_cars"], prof["ev_trucks"]]

    for tab, data, ylabel in zip(tabs[:4], plots, ["kW","kW","kW","kW"]):
        with tab:
            if plt is None:
                st.line_chart(data.rename(ylabel))
            else:
                fig, ax = plt.subplots()
                data.plot(ax=ax)
                ax.set_ylabel(ylabel)
                st.pyplot(fig)

    with tabs[4]:
        if plt is None:
            st.line_chart(soc.rename("SOC (kWh)"))
            st.line_chart(grid_imp_series.rename("Grid (kW)"))
        else:
            fig, ax = plt.subplots()
            soc.plot(ax=ax, label="SOC (kWh)")
            ax2 = ax.twinx()
            grid_imp_series.plot(ax=ax2, label="Grid (kW)")
            ax.set_ylabel("kWh"); ax2.set_ylabel("kW")
            ax.legend(loc="upper left"); ax2.legend(loc="upper right")
            st.pyplot(fig)

    with tabs[5]:
        if plt is None:
            st.line_chart(residual.rename("Residual (kW) = Load + EV - PV"))
        else:
            fig, ax = plt.subplots()
            residual.plot(ax=ax)
            ax.set_ylabel("kW")
            ax.set_title("Residual (kW) = Last + EV − PV")
            st.pyplot(fig)

    # ---- Download SOC ----
    st.subheader("SOC‑CSV herunterladen")
    buf = io.StringIO()
    soc.to_csv(buf)
    st.download_button("Download CSV", buf.getvalue(), file_name="battery_soc.csv")
