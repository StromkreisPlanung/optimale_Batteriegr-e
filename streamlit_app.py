from __future__ import annotations
"""
streamlit_app.py · Battery‑Sizing Dashboard  v0.9.0
===================================================
• CSV deutsch/englisch (Semikolon/Komma, Dezimalkomma/-punkt) + DST-Fix
• Mehrere PV‑CSV (aufsummiert)
• Smart‑EV (priorisiert PV‑Überschuss)
• PyPSA mit tz‑naiven Snapshots + korrekter Zeitgewichtung (h)
• PV als positive Erzeugung, optional mit Einspeisevergütung (neg. marginal_cost)
• Batterie‑CAPEX: annuitätisch (Lebensdauer,WACC) und auf Datenlaufzeit skaliert
• Netzimport: konstanter €/kWh ODER zeitvariable Preise (CSV)
• Demand‑Charge (€/kW/Monat) als KPI (ex‑post)
• Residual‑Fix: Duplikate vor reindex entfernen
• Optional: Sensitivität zu max_hours (unverändert)
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
def _read_any_csv(upload) -> pd.DataFrame:
    """Liest CSV mit Komma oder Semikolon, Dezimalpunkt/‑komma; gibt DataFrame zurück."""
    raw = upload.getvalue().decode("utf-8-sig")
    header = raw.splitlines()[0] if raw else ""
    semicolon = ";" in header
    sep, dec = (";", ",") if semicolon else (",", ".")
    df = pd.read_csv(io.StringIO(raw), sep=sep, decimal=dec, engine="python")
    return df

def read_profile(upload, res: str) -> pd.Series:
    """Erwartet Spalten: datetime, power_kw (Trennzeichen/Dezimal auto)."""
    df = _read_any_csv(upload)
    if not {"datetime", "power_kw"}.issubset({c.strip().lower() for c in df.columns}):
        # Toleranter Mapper
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        dt_col = lower_map.get("datetime") or lower_map.get("zeit") or lower_map.get("date") or lower_map.get("datum")
        p_col  = lower_map.get("power_kw") or lower_map.get("leistung") or lower_map.get("kw") or lower_map.get("power")
        if not dt_col or not p_col:
            raise ValueError("CSV benötigt Spalten 'datetime' und 'power_kw'.")
        df = df.rename(columns={dt_col: "datetime", p_col: "power_kw"})
    else:
        # exakt gleich
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "datetime", cols[1]: "power_kw"})

    # Parse datetime (deutsch/ISO)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    # Zahlen: Komma/Punkt bereits vom read_csv berücksichtigt; trotzdem coercen:
    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
    df = df.dropna(subset=["datetime", "power_kw"])

    # Doppelte Timestamps mitteln & resampeln
    df = df.groupby("datetime", as_index=False)["power_kw"].mean().set_index("datetime")
    s = df["power_kw"].resample(res).mean().fillna(0.0)

    # TZ anwenden (DST), dann wieder tz-naiv für PyPSA
    s.index = s.index.tz_localize("Europe/Vienna", nonexistent="shift_forward", ambiguous=False)
    s.index = s.index.tz_localize(None)
    return s

def read_price_series(upload, res: str, colname: str = "price_eur_per_kwh") -> pd.Series:
    """CSV mit Spalten: datetime, price_eur_per_kwh (de/intl)."""
    df = _read_any_csv(upload)
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    dt_col = lower_map.get("datetime") or lower_map.get("zeit") or lower_map.get("date") or lower_map.get("datum")
    p_col  = lower_map.get(colname) or lower_map.get("preis") or lower_map.get("price") or lower_map.get("preis_eur_kwh")
    if not dt_col or not p_col:
        raise ValueError("Preis-CSV benötigt Spalten 'datetime' und 'price_eur_per_kwh'.")
    df = df.rename(columns={dt_col: "datetime", p_col: "price"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["datetime", "price"]).groupby("datetime", as_index=False)["price"].mean().set_index("datetime")
    s = df["price"].resample(res).mean().fillna(method="ffill")
    s.index = s.index.tz_localize("Europe/Vienna", nonexistent="shift_forward", ambiguous=False)
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
    """e_kwh/Tag im Fenster, zuerst PV‑Überschusszeiten (residual<0), dann Rest."""
    out = pd.Series(0.0, index=load_kw.index)
    h = (out.index[1] - out.index[0]).total_seconds() / 3600.0
    true_idx = mask.index[mask]

    for _, grp in out.groupby(out.index.date):
        idx = grp.index[grp.index.isin(true_idx)]
        if len(idx) == 0:
            continue
        residual = (load_kw[idx] - pv_kw[idx])
        surplus = (-residual).clip(lower=0.0)
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

# ---------- Sanitizer für Reindex/Plots (duplikat-sicher) ---------- #
def _sanitize_for_snaps(s: pd.Series, snaps: pd.DatetimeIndex) -> pd.Series:
    s = s.groupby(level=0).mean().sort_index()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.reindex(snaps).fillna(0.0)

# ----------------- Annuität & Zeitanteil ----------------- #
def annuity(r: float, n: int) -> float:
    return (r*(1+r)**n)/((1+r)**n - 1) if r > 0 else 1.0/n

def years_covered_from_snapshots(snaps: pd.DatetimeIndex) -> float:
    if len(snaps) >= 2:
        dt_h = (snaps[1] - snaps[0]).total_seconds() / 3600.0
    else:
        dt_h = 1.0
    return float(len(snaps) * dt_h) / 8760.0

# ------------------- PyPSA‑Netz (mit Preisen) -------------- #
def build_network(p: dict[str, pd.Series],
                  grid_kw: float,
                  cap_kwh: float,
                  cap_kw: float,
                  h_batt: float,
                  buy_price: float | pd.Series,
                  pv_feed_in: float,
                  life: int,
                  wacc: float) -> pypsa.Network:
    """Zeitreihen defensiv, Snapshots + Zeitgewichtung (h), Netz‑Preis (konstant/Serie), PV‑Vergütung, CAPEX annuitätisch skaliert."""
    n = pypsa.Network()
    snaps = pd.Index(p["load"].index).drop_duplicates(keep="first").sort_values()
    n.set_snapshots(snaps)

    # Zeitgewichtung je Snapshot (h)
    if len(snaps) >= 2:
        dt_h = (snaps[1] - snaps[0]).total_seconds() / 3600.0
    else:
        dt_h = 1.0
    n.snapshot_weightings = pd.Series(dt_h, index=snaps)

    n.add("Bus", "grid")

    # Lasten
    n.add("Load", "demand", bus="grid")
    n.add("Load", "ev_cars", bus="grid")
    n.add("Load", "ev_trucks", bus="grid")

    # PV als variable Erzeugung mit Obergrenze (= Profil) und optionaler Vergütung:
    # p_nom=1, p_max_pu = pv_kW -> maximaler Output = pv_kW
    n.add("Generator", "pv", bus="grid", p_nom=1.0, marginal_cost=-pv_feed_in)
    # Sanitized Zeitreihen exakt auf snaps
    load_s   = _sanitize_for_snaps(p["load"], snaps)
    ev_cars  = _sanitize_for_snaps(p["ev_cars"], snaps)
    ev_truck = _sanitize_for_snaps(p["ev_trucks"], snaps)
    pv_s     = _sanitize_for_snaps(p["pv"], snaps)

    n.loads_t.p_set = pd.DataFrame(index=snaps, data={
        "demand":    load_s.values,
        "ev_cars":   ev_cars.values,
        "ev_trucks": ev_truck.values,
    })
    # PV‑Verfügbarkeit als p_max_pu
    n.generators_t.p_max_pu = pd.DataFrame(index=snaps, data={"pv": pv_s.values})
    # (Hinweis: p_nom=1.0 ⇒ Obergrenze = p_max_pu selbst in kW)

    # Netz‑Import als dispatchbarer Generator (Preis konstant ODER Serie)
    n.add("Generator", "grid_import", bus="grid", p_nom=grid_kw, p_nom_extendable=False, p_max_pu=1.0, p_min_pu=0.0)
    if isinstance(buy_price, pd.Series):
        bp = _sanitize_for_snaps(buy_price, snaps)
        n.generators_t.marginal_cost = pd.DataFrame(index=snaps, data={"grid_import": bp.values})
    else:
        n.generators.loc["grid_import", "marginal_cost"] = float(buy_price)

    # Batterie: feste Dauer (max_hours) & annuitätisch/zeitanteilig bewertete CAPEX
    cap_eff_total = cap_kw + cap_kwh * h_batt          # €/kW (Turnkey)
    ann = annuity(wacc, life)                          # 1/a
    years_cov = years_covered_from_snapshots(snaps)    # a im Datensatz
    cap_cost_model = cap_eff_total * ann * years_cov   # €/kW für Zeitraum

    n.add("StorageUnit", "battery",
          bus="grid",
          p_nom_extendable=True,
          max_hours=h_batt,
          efficiency_store=0.95,
          efficiency_dispatch=0.95,
          capital_cost=cap_cost_model,
          marginal_cost=0.0)
    return n

# --------------- Sensitivität: Hilfsfunktion --------------- #
def run_sensitivity(prof: dict,
                    grid_kw: float,
                    cap_kwh: float,
                    cap_kw: float,
                    price_buy: float | pd.Series,
                    pv_feed_in: float,
                    life: int,
                    wacc: float,
                    hours_list: list[float]) -> pd.DataFrame:
    rows = []
    prog = st.progress(0.0)
    total = max(1, len(hours_list))
    for i, h in enumerate(hours_list, start=1):
        p = float("nan")
        try:
            n = build_network(prof, grid_kw, cap_kwh, cap_kw, h, price_buy, pv_feed_in, life, wacc)
            n.optimize()
            p = float(n.storage_units.loc["battery", "p_nom_opt"])
        except Exception:
            p = float("nan")

        e = p * h if pd.notna(p) else float("nan")
        cap_eff = cap_kw + cap_kwh * h
        # Annuität & Zeitraum
        snaps = pd.Index(prof["load"].index).drop_duplicates().sort_values()
        years_cov = years_covered_from_snapshots(snaps)
        ann = annuity(wacc, life)
        invest = (cap_eff * ann * years_cov * p) if pd.notna(p) else float("nan")

        rows.append({
            "max_hours_h": h,
            "p_nom_kw": p,
            "e_nom_kwh": e,
            "capital_cost_eur_per_kw_eff": cap_eff,
            "invest_estimate_eur_for_period": invest,
        })
        prog.progress(min(1.0, i / total))

    df = pd.DataFrame(rows).sort_values("max_hours_h").set_index("max_hours_h")
    return df

# ----------------------- UI ------------------------------- #
st.title("🔋 Optimale Batteriegröße bestimmen")

sb = st.sidebar
sb.header("Basisparameter")
res       = sb.selectbox("Zeitauflösung", ["15min", "60min"], 0)
grid_kw   = sb.number_input("Grid‑Anschluss (kW)", min_value=10, value=800, step=10)
cap_kwh   = sb.number_input("CapEx €/kWh", min_value=0.0, value=300.0, step=10.0)
cap_kw    = sb.number_input("CapEx €/kW",  min_value=0.0, value=150.0, step=10.0)
h_batt    = sb.number_input("Batteriedauer max_hours (h)", min_value=0.25, max_value=12.0, value=2.0, step=0.25)
life      = sb.number_input("Lebensdauer Batterie (a)", min_value=1, max_value=25, value=10)
wacc      = sb.number_input("WACC / Zins (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.5) / 100.0

sb.header("Energiepreise")
price_buy_const = sb.number_input("Strompreis Netz (€/kWh)", min_value=0.0, value=0.25, step=0.01, format="%.2f")
price_file = sb.file_uploader("Optional: Zeitvariable Preise (CSV: datetime, price_eur_per_kwh)", type=["csv"])

sb.header("PV‑Erlöse & Demand‑Charge")
pv_feed_in = sb.number_input("Einspeisevergütung PV (€/kWh)", min_value=0.0, value=0.00, step=0.01, format="%.2f")
demand_charge = sb.number_input("Demand‑Charge (€/kW/Monat)", min_value=0.0, value=0.00, step=1.0, format="%.2f")

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

    price_series = None
    if price_file is not None:
        try:
            price_series = read_price_series(price_file, res)
        except Exception as e:
            st.error(f"Preis‑CSV fehlerhaft: {e}")
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
    buy_price_input = price_series if price_series is not None else float(price_buy_const)
    net = build_network(prof, grid_kw, cap_kwh, cap_kw, h_batt, buy_price_input, pv_feed_in, life, wacc)

    with st.spinner("Optimierung läuft …"):
        net.optimize()

    # Ergebnisse
    p_nom = float(net.storage_units.loc["battery", "p_nom_opt"])
    e_nom = p_nom * h_batt

    # Netz‑Import‑Energie und Kosten
    gi = net.generators_t["p"].get("grid_import")
    if len(net.snapshots) >= 2:
        dt_h = (net.snapshots[1] - net.snapshots[0]).total_seconds() / 3600.0
    else:
        dt_h = 1.0
    energy_kwh = float((gi * dt_h).sum()) if gi is not None else 0.0
    if price_series is not None:
        ps = _sanitize_for_snaps(price_series, net.snapshots)
        energy_cost_eur = float((gi * dt_h * ps).sum())
    else:
        energy_cost_eur = energy_kwh * float(price_buy_const)

    # Demand‑Charge KPI (ex‑post): max(Import) je Monat * €/kW/Monat
    if gi is not None and demand_charge > 0:
        gi_month = gi.copy()
        gi_month.index = pd.to_datetime(gi_month.index)
        monthly_peak = gi_month.resample("MS").max()  # Monatsanfang-Index
        demand_cost = float((monthly_peak * demand_charge).sum())
    else:
        demand_cost = 0.0

    # PV‑Erzeugung (tatsächlich) & PV‑Vergütungswert (nur für erzeugte kWh, kein expliziter Exportpfad)
    pv_generated = net.generators_t["p"].get("pv")
    pv_energy_kwh = float((pv_generated * dt_h).sum()) if pv_generated is not None else 0.0
    pv_value_eur = pv_energy_kwh * float(pv_feed_in) if pv_feed_in > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Energie (kWh)", f"{e_nom:.1f}")
    c2.metric("Leistung (kW)", f"{p_nom:.1f}")
    c3.metric("Netzbezug / Kosten", f"{energy_kwh:,.0f} kWh / {energy_cost_eur:,.0f} €")

    if demand_charge > 0 or pv_feed_in > 0:
        st.info(f"Demand‑Charge (ex‑post): {demand_cost:,.0f} €  ·  PV‑Wert (Erzeugung×Vergütung): {pv_value_eur:,.0f} €")

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

    # Residual (Last + EV – PV) duplikat-sicher und auf snaps reindext
    residual = _sanitize_for_snaps(prof["load"] + prof["ev_cars"] + prof["ev_trucks"] - prof["pv"], net.snapshots)

    # ---- Plots ----
    st.subheader("Zeitreihen")
    tabs = st.tabs(["Load", "PV avail", "PV gen", "EV Cars", "EV Trucks", "SOC + Grid", "Residual"])
    plots = [prof["load"], prof["pv"], pv_generated if pv_generated is not None else pd.Series(0.0, index=net.snapshots),
             prof["ev_cars"], prof["ev_trucks"]]

    for tab, data, ylabel in zip(tabs[:5], plots, ["kW","kW","kW","kW","kW"]):
        with tab:
            series = _sanitize_for_snaps(data, net.snapshots) if not isinstance(data, pd.Series) or not data.index.equals(net.snapshots) else data
            if plt is None:
                st.line_chart(series.rename(ylabel))
            else:
                fig, ax = plt.subplots()
                series.plot(ax=ax)
                ax.set_ylabel(ylabel)
                st.pyplot(fig)

    with tabs[5]:
        grid_imp_series = gi if gi is not None else pd.Series(0.0, index=net.snapshots)
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

    with tabs[6]:
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

    # --------------- Sensitivitätsanalyse: UI --------------- #
    st.divider()
    st.subheader("Sensitivität zu max_hours")

    sens_on = st.checkbox("Sensitivitätsanalyse aktivieren (mehrere Batteriedauern testen)", value=False)

    if sens_on:
        colA, colB, colC = st.columns(3)
        with colA:
            h_min = st.number_input("min (h)", min_value=0.25, max_value=12.0, value=1.0, step=0.25)
        with colB:
            h_max = st.number_input("max (h)", min_value=0.25, max_value=12.0, value=4.0, step=0.25)
        with colC:
            h_step = st.number_input("Schritt (h)", min_value=0.25, max_value=4.0, value=0.5, step=0.25)

        hours_list = []
        h = h_min
        while h <= h_max + 1e-9:
            hours_list.append(round(h, 2))
            h += h_step

        if st.button("⏱️ Sensitivität rechnen"):
            with st.spinner("Berechne Sensitivität…"):
                df_sens = run_sensitivity(prof, grid_kw, cap_kwh, cap_kw, buy_price_input, pv_feed_in, life, wacc, hours_list)

            st.write("**Ergebnisse (je Dauer):**")
            st.dataframe(
                df_sens.style.format({
                    "p_nom_kw": "{:.1f}",
                    "e_nom_kwh": "{:.1f}",
                    "capital_cost_eur_per_kw_eff": "{:.0f}",
                    "invest_estimate_eur_for_period": "{:,.0f}",
                }),
                use_container_width=True
            )

            # Kurven zeichnen
            if plt is None:
                st.line_chart(df_sens[["p_nom_kw"]].rename(columns={"p_nom_kw": "p_nom (kW)"}))
                st.line_chart(df_sens[["e_nom_kwh"]].rename(columns={"e_nom_kwh": "e_nom (kWh)"}))
                st.line_chart(df_sens[["invest_estimate_eur_for_period"]].rename(columns={"invest_estimate_eur_for_period": "Invest (EUR, Zeitraum)"}))
            else:
                for col, ylabel, title in [
                    ("p_nom_kw", "kW", "Optimale Leistung vs. max_hours"),
                    ("e_nom_kwh", "kWh", "Optimale Energie vs. max_hours"),
                    ("invest_estimate_eur_for_period", "EUR", "Invest‑Schätzung (Zeitraum) vs. max_hours"),
                ]:
                    fig, ax = plt.subplots()
                    df_sens[col].plot(ax=ax, marker="o")
                    ax.set_xlabel("max_hours (h)")
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    st.pyplot(fig)
