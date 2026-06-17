"""
Solar PV Time Series & Energy Analysis Tool
Data: NASA POWER (CERES/SRB) · Calculations: pvlib
"""

import io
import datetime
import warnings
import requests
import numpy as np
import pandas as pd
import pvlib
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium
from timezonefinder import TimezoneFinder

warnings.filterwarnings("ignore")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar PV Analysis Tool",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
_START_YEAR_DEFAULT = 2015
_END_YEAR_DEFAULT   = 2023
_MIN_YEAR = 1990
_MAX_YEAR = 2023

NASA_URL    = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_PARAMS = "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,CLRSKY_SFC_SW_DWN,T2M,WS2M"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
section[data-testid="stSidebar"] { background:#F8FAFC; border-right:1px solid #E2E8F0; }
section[data-testid="stSidebar"] .stButton>button {
    background:#1B3A6B; color:#FFF; border-radius:6px; font-weight:600;
    width:100%; padding:0.6rem 1rem; border:none;
}
section[data-testid="stSidebar"] .stButton>button:hover { background:#254e94; }
.kpi-card {
    background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;
    padding:1rem 0.8rem; text-align:center;
}
.kpi-card .val  { font-size:1.55rem; font-weight:700; color:#1B3A6B; line-height:1.1; }
.kpi-card .unit { font-size:0.72rem; color:#64748B; margin-top:0.1rem; }
.kpi-card .lbl  { font-size:0.78rem; color:#475569; margin-top:0.2rem; }
.stDownloadButton>button {
    background:#0F766E; color:#FFF; border-radius:6px;
    font-weight:600; width:100%; border:none;
}
.stDownloadButton>button:hover { background:#0d6460; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1B3A6B 0%,#254e94 100%);
            border-radius:10px; padding:1.2rem 1.8rem; margin-bottom:1.2rem;">
  <h1 style="font-size:1.7rem; font-weight:700; color:#FFF; margin:0;
             letter-spacing:-0.03em; line-height:1.2;">
    Solar PV Time Series &amp; Energy Analysis<br>
    <span style="font-size:1.05rem; font-weight:400; color:#BFD3F5;">
      NASA POWER (CERES/SRB) · pvlib · Hourly UTC
    </span>
  </h1>
</div>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_tz(lat: float, lon: float) -> str:
    return TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC"


@st.cache_data(show_spinner=False)
def _get_elevation(lat: float, lon: float) -> int:
    """Auto-fetch elevation (m ASL) from Open-Topo-Data SRTM 30m dataset."""
    try:
        r = requests.get(
            "https://api.opentopodata.org/v1/srtm30m",
            params={"locations": f"{lat},{lon}"},
            timeout=10,
        )
        r.raise_for_status()
        elev = r.json()["results"][0]["elevation"]
        return int(round(elev)) if elev is not None else 0
    except Exception:
        return 0


def _fetch_nasa_year(lat: float, lon: float, year: int) -> pd.DataFrame:
    """Fetch one year of hourly data from NASA POWER (no time-standard param for hourly)."""
    resp = requests.get(NASA_URL, params={
        "parameters": NASA_PARAMS,
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      f"{year}0101",
        "end":        f"{year}1231",
        "format":     "JSON",
    }, timeout=120)
    if not resp.ok:
        raise RuntimeError(
            f"NASA POWER API error {resp.status_code} for year {year}: {resp.text[:300]}"
        )
    param_data = resp.json()["properties"]["parameter"]
    first_key  = next(iter(param_data))
    timestamps = list(param_data[first_key].keys())
    df = pd.DataFrame(param_data, index=timestamps)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    # NASA POWER hourly data is in Local Solar Time (LST), not UTC.
    # Localize with a fixed UTC offset = round(lon/15) hours so that pvlib
    # and all downstream timezone conversions see the correct absolute time.
    lst_tz = datetime.timezone(datetime.timedelta(hours=round(lon / 15.0)))
    df.index = df.index.tz_localize(lst_tz)
    return df


@st.cache_data(show_spinner=False)
def fetch_nasa_power(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch hourly GHI/DNI/DHI/T2M/WS2M from NASA POWER; returns UTC DataFrame."""
    frames = []
    for yr in range(start_year, end_year + 1):
        frames.append(_fetch_nasa_year(lat, lon, yr))
    df = pd.concat(frames).sort_index()

    df.rename(columns={
        "ALLSKY_SFC_SW_DWN":  "ghi",
        "ALLSKY_SFC_SW_DNI":  "dni",
        "ALLSKY_SFC_SW_DIFF": "dhi",
        "CLRSKY_SFC_SW_DWN":  "ghi_clear",
        "T2M":   "temp_air",
        "WS2M":  "wind_speed",
    }, inplace=True)

    df.replace(-999, np.nan, inplace=True)
    df.replace(-999.0, np.nan, inplace=True)
    df = df.astype(float)

    for c in ["ghi", "dni", "dhi", "ghi_clear"]:
        df[c] = df[c].clip(lower=0)
    for c in ["temp_air", "wind_speed"]:
        df[c] = df[c].ffill().fillna(15.0 if c == "temp_air" else 1.0)

    return df


def run_solar_pipeline(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    elevation: float,
    tz: str,
    tilt: float,
    azimuth: float,
    capacity_kwp: float,
    dc_ac_ratio: float,
    temp_coeff: float,
    system_losses_pct: float,
) -> tuple:
    """pvlib pipeline → (result_df, meta_dict)."""
    location = pvlib.location.Location(lat, lon, tz=tz, altitude=elevation)
    times_local = df.index.tz_convert(tz)
    solpos = location.get_solarposition(times_local)

    # Extraterrestrial DNI (for Hay-Davies transposition model)
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)

    # Plane-of-array irradiance (Hay-Davies: separates isotropic/circumsolar/horizon)
    poa_irr = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"].values,
        solar_azimuth=solpos["azimuth"].values,
        dni=df["dni"].values,
        ghi=df["ghi"].values,
        dhi=df["dhi"].values,
        dni_extra=dni_extra.values,
        model="haydavies",
    )
    poa = pd.Series(poa_irr["poa_global"], index=df.index).clip(lower=0).fillna(0)

    # Cell temperature — Faiman model (accounts for wind cooling)
    # T_cell = T_air + G_poa / (U0 + U1 × wind_speed)  [default U0=25, U1=6.84]
    temp_cell_arr = pvlib.temperature.faiman(
        poa_global=poa.values,
        temp_air=df["temp_air"].values,
        wind_speed=df["wind_speed"].values,
    )
    temp_cell = pd.Series(temp_cell_arr, index=df.index)

    # DC power: P_dc = C_kwp × (G_poa/1000) × [1 + (γ/100) × (T_cell − 25)]
    dc_power = capacity_kwp * (poa / 1000.0) * (1.0 + (temp_coeff / 100.0) * (temp_cell - 25.0))
    dc_power = dc_power.clip(lower=0)

    # AC power: clipped at inverter rated capacity
    inverter_kw = capacity_kwp / dc_ac_ratio
    ac_power = dc_power.clip(upper=inverter_kw)

    # System losses (soiling, wiring, mismatch, availability…)
    ac_power = (ac_power * (1.0 - system_losses_pct / 100.0)).clip(lower=0)

    # Cloud shading: fraction of clear-sky GHI lost to cloud cover
    # Only meaningful during daylight (clear-sky GHI > 5 W/m²)
    ghi_clear = df["ghi_clear"]
    cloud_transmittance = np.where(ghi_clear > 5, df["ghi"] / ghi_clear, np.nan)
    cloud_transmittance = np.clip(cloud_transmittance, 0.0, 1.0)
    cloud_shading_pct   = pd.Series((1.0 - cloud_transmittance) * 100, index=df.index)

    result = pd.DataFrame({
        "ghi_wm2":          df["ghi"].round(1),
        "ghi_clear_wm2":    ghi_clear.round(1),
        "cloud_shading_pct": cloud_shading_pct.round(1),
        "dni_wm2":          df["dni"].round(1),
        "dhi_wm2":          df["dhi"].round(1),
        "poa_wm2":          poa.round(1),
        "temp_air_c":       df["temp_air"].round(2),
        "temp_cell_c":      temp_cell.round(2),
        "wind_speed_ms":    df["wind_speed"].round(2),
        "dc_power_kw":      dc_power.round(4),
        "ac_power_kw":      ac_power.round(4),
    }, index=df.index)

    n_years = (df.index[-1] - df.index[0]).days / 365.25
    ann_ghi  = df["ghi"].sum() / 1000.0 / n_years
    ann_poa  = poa.sum() / 1000.0 / n_years
    ann_ac   = ac_power.sum() / n_years
    sy       = ann_ac / capacity_kwp
    cf       = ann_ac / (inverter_kw * 8760.0)
    pr       = ann_ac / (ann_ghi * capacity_kwp)
    mean_cloud_shading = float(np.nanmean(cloud_transmittance))
    mean_cloud_shading = (1.0 - mean_cloud_shading) * 100   # % of clear-sky lost

    meta = {
        "n_years": n_years,
        "annual_ghi_kwh_m2":   ann_ghi,
        "annual_poa_kwh_m2":   ann_poa,
        "annual_ac_kwh":       ann_ac,
        "specific_yield":      sy,
        "capacity_factor":     cf,
        "performance_ratio":   pr,
        "inverter_kw":         inverter_kw,
        "mean_cloud_shading_pct": mean_cloud_shading,
    }
    return result, meta


# ── chart helpers ─────────────────────────────────────────────────────────────
_NAVY  = "#1B3A6B"
_TEAL  = "#0F766E"
_AMBER = "#D97706"
_SLATE = "#64748B"


def _chart_monthly_energy(result_df: pd.DataFrame, tz: str) -> go.Figure:
    local = result_df["ac_power_kw"].copy()
    local.index = local.index.tz_convert(tz)
    avg = local.groupby([local.index.year, local.index.month]).sum().groupby(level=1).mean() / 1000.0
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Bar(
        x=months, y=avg.values.round(1),
        marker_color=_NAVY,
        text=[f"{v:.0f}" for v in avg.values], textposition="outside",
    ))
    fig.update_layout(
        title="Average Monthly AC Energy Output",
        yaxis_title="MWh", height=310,
        margin=dict(l=50,r=20,t=40,b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def _chart_daily_profile(result_df: pd.DataFrame, tz: str) -> go.Figure:
    local = result_df["ac_power_kw"].copy()
    local.index = local.index.tz_convert(tz)
    df2 = pd.DataFrame({"kw": local.values, "hour": local.index.hour, "month": local.index.month})
    # Southern-hemisphere seasons by month
    season_map = {12:"Summer",1:"Summer",2:"Summer",
                  3:"Autumn",4:"Autumn",5:"Autumn",
                  6:"Winter",7:"Winter",8:"Winter",
                  9:"Spring",10:"Spring",11:"Spring"}
    df2["season"] = df2["month"].map(season_map)
    colors = {"Summer":"#EF4444","Autumn":"#D97706","Winter":"#3B82F6","Spring":"#22C55E"}
    fig = go.Figure()
    for s, c in colors.items():
        grp = df2[df2["season"] == s].groupby("hour")["kw"].mean()
        fig.add_trace(go.Scatter(x=grp.index, y=grp.values.round(1), name=s,
                                 line=dict(color=c, width=2)))
    fig.update_layout(
        title="Average Diurnal AC Power Profile (by Season)",
        xaxis=dict(title="Hour (local time)", dtick=2, range=[0,23]),
        yaxis_title="kW", height=310,
        margin=dict(l=50,r=20,t=40,b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#CBD5E1")
    fig.update_xaxes(gridcolor="#E2E8F0")
    return fig


def _chart_irradiance(result_df: pd.DataFrame) -> go.Figure:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    grp = result_df.groupby(result_df.index.month)
    ghi_m      = grp["ghi_wm2"].mean().round(1)
    ghi_clr_m  = grp["ghi_clear_wm2"].mean().round(1)
    poa_m      = grp["poa_wm2"].mean().round(1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=ghi_clr_m.values, name="Clear-sky GHI",
                             line=dict(color="#CBD5E1", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=months, y=ghi_m.values, name="GHI (all-sky)",
                             line=dict(color=_SLATE, width=2)))
    fig.add_trace(go.Scatter(x=months, y=poa_m.values, name="POA (tilted)",
                             line=dict(color=_AMBER, width=2)))
    fig.update_layout(
        title="Monthly Mean Irradiance — Clear-sky / GHI / POA",
        yaxis_title="W/m²", height=310,
        margin=dict(l=50,r=20,t=40,b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def _chart_cloud_shading(result_df: pd.DataFrame) -> go.Figure:
    """Monthly average daytime cloud shading (% of clear-sky GHI lost)."""
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    # Only use daytime rows (clear-sky GHI > 5 W/m²)
    day = result_df[result_df["ghi_clear_wm2"] > 5]
    shading_m = day.groupby(day.index.month)["cloud_shading_pct"].mean().round(1)
    fig = go.Figure(go.Bar(
        x=months, y=shading_m.values,
        marker_color=_SLATE,
        text=[f"{v:.0f}%" for v in shading_m.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Monthly Average Cloud Shading (% of Clear-Sky GHI Lost)",
        yaxis_title="Cloud Shading (%)", height=310,
        margin=dict(l=50,r=20,t=40,b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_range=[0, max(shading_m.values) * 1.25],
    )
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def _chart_annual_cf(result_df: pd.DataFrame, inverter_kw: float) -> go.Figure:
    ann = result_df.groupby(result_df.index.year)["ac_power_kw"].sum()
    cf  = (ann / (inverter_kw * 8760.0) * 100).round(1)
    fig = go.Figure(go.Bar(
        x=cf.index.astype(str), y=cf.values,
        marker_color=_TEAL,
        text=[f"{v:.1f}%" for v in cf.values], textposition="outside",
    ))
    fig.update_layout(
        title="Annual AC Capacity Factor by Year",
        yaxis_title="Capacity Factor (%)", height=280,
        margin=dict(l=50,r=20,t=40,b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False,
                     range=[0, max(cf.values) * 1.25])
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📍 Location")
    lat = st.number_input("Latitude",  value=-31.9505, min_value=-90.0,  max_value=90.0,  step=0.001, format="%.4f")
    lon = st.number_input("Longitude", value=115.8605, min_value=-180.0, max_value=180.0, step=0.001, format="%.4f")
    elev_auto = _get_elevation(lat, lon)
    elevation = st.number_input(
        "Elevation (m ASL)", value=elev_auto,
        min_value=0, max_value=9000, step=1,
        key=f"elev_{lat:.4f}_{lon:.4f}",
        help="Auto-fetched from SRTM 30m data. Override if needed.",
    )

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    st.caption(f"NASA POWER data: 1990 – {_MAX_YEAR}")
    start_year = st.number_input("Start Year", value=_START_YEAR_DEFAULT,
                                 min_value=_MIN_YEAR, max_value=_MAX_YEAR, step=1)
    end_year   = st.number_input("End Year",   value=_END_YEAR_DEFAULT,
                                 min_value=_MIN_YEAR, max_value=_MAX_YEAR, step=1)
    if end_year < start_year:
        st.warning("End year must be ≥ start year.")
        end_year = start_year

    st.markdown("---")
    st.markdown("### ⚙️ System Parameters")

    capacity_kwp = st.number_input(
        "System Capacity (kWp DC)", value=10_000.0,
        min_value=1.0, max_value=2_000_000.0, step=500.0,
        help="Peak DC array capacity.",
    )

    tilt_default = min(int(round(abs(lat))), 60)
    tilt = st.slider(
        "Panel Tilt (° from horizontal)", 0, 90, tilt_default,
        help="Optimal ≈ site latitude for maximum annual yield.",
    )

    az_default = 0   # North-facing for southern hemisphere
    azimuth = st.slider(
        "Panel Azimuth (° from North)", 0, 359, az_default,
        help="0° = North (optimal S. hemisphere) · 180° = South (optimal N. hemisphere)",
    )

    dc_ac_ratio = st.slider(
        "DC:AC Ratio", 1.0, 1.5, 1.15, 0.05,
        help="DC capacity ÷ inverter AC rating. Typical: 1.1–1.3.",
    )

    temp_coeff = st.slider(
        "Temperature Coefficient (%/°C)", -0.60, -0.20, -0.40, 0.01,
        help="Power loss per °C above 25 °C STC. Mono-Si typical: −0.35 to −0.45 %/°C.",
    )

    system_losses = st.slider(
        "System Losses (%)", 0, 30, 8,
        help="Soiling, wiring, inverter, mismatch, availability. Typical: 5–15%.",
    )

    st.markdown("---")
    run_btn = st.button("☀️  Run Analysis", use_container_width=True)

# ── map + site info ───────────────────────────────────────────────────────────
col_map, col_info = st.columns([3, 2])

with col_map:
    m = folium.Map(location=[lat, lon], zoom_start=8, tiles="CartoDB positron")
    folium.Marker(
        [lat, lon],
        tooltip=f"{lat:.4f}, {lon:.4f}",
        icon=folium.Icon(color="orange", icon="sun-o", prefix="fa"),
    ).add_to(m)
    st_folium(m, height=290, width=None, returned_objects=[])

with col_info:
    tz_auto = _get_tz(lat, lon)
    inv_kw  = capacity_kwp / dc_ac_ratio
    st.markdown(f"""
**Site**
Lat / Lon: `{lat:.4f}° / {lon:.4f}°`
Elevation: `{elevation} m ASL` · Timezone: `{tz_auto}`

**System**
DC Capacity: `{capacity_kwp:,.0f} kWp`
Tilt / Azimuth: `{tilt}° / {azimuth}° (from N)`
DC:AC ratio: `{dc_ac_ratio:.2f}` → Inverter: `{inv_kw:,.0f} kW AC`
Temp coefficient: `{temp_coeff:.2f} %/°C`
System losses: `{system_losses} %`

**Data source**
NASA POWER · CERES/SRB · Hourly UTC
Period: `{start_year} – {end_year}`
    """)

st.markdown("---")

# ── run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    tz = _get_tz(lat, lon)

    n_years_req = end_year - start_year + 1
    with st.spinner(f"Fetching NASA POWER hourly data ({start_year}–{end_year}, {n_years_req} API calls)…"):
        try:
            df_raw = fetch_nasa_power(lat, lon, start_year, end_year)
        except Exception as e:
            st.error(f"NASA POWER fetch failed: {e}")
            st.stop()

    with st.spinner(f"Running pvlib solar pipeline ({len(df_raw):,} records)…"):
        result_df, meta = run_solar_pipeline(
            df_raw, lat, lon, elevation, tz,
            tilt, azimuth, capacity_kwp, dc_ac_ratio, temp_coeff, system_losses,
        )

    st.session_state.update({
        "solar_result": result_df,
        "solar_meta":   meta,
        "solar_params": {
            "lat": lat, "lon": lon, "elevation": elevation, "tz": tz,
            "capacity_kwp": capacity_kwp, "tilt": tilt, "azimuth": azimuth,
            "dc_ac_ratio": dc_ac_ratio, "temp_coeff": temp_coeff,
            "system_losses": system_losses,
            "start_year": start_year, "end_year": end_year,
        },
    })

# ── results ───────────────────────────────────────────────────────────────────
if "solar_result" in st.session_state:
    result_df = st.session_state["solar_result"]
    meta      = st.session_state["solar_meta"]
    p         = st.session_state["solar_params"]

    n_yr   = meta["n_years"]
    aey    = meta["annual_ac_kwh"]              # kWh/yr
    sy     = meta["specific_yield"]             # kWh/kWp/yr
    cf     = meta["capacity_factor"] * 100      # %
    pr     = meta["performance_ratio"] * 100    # %
    ghi    = meta["annual_ghi_kwh_m2"]          # kWh/m2/yr
    poa    = meta["annual_poa_kwh_m2"]          # kWh/m2/yr
    inv    = meta["inverter_kw"]
    cloud  = meta["mean_cloud_shading_pct"]     # %

    # ── KPI cards ──────────────────────────────────────────────────────────────
    def _kpi(col, val, unit, label):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="val">{val}</div>'
            f'<div class="unit">{unit}</div>'
            f'<div class="lbl">{label}</div>'
            f'</div>', unsafe_allow_html=True
        )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    _kpi(c1, f"{aey/1000:,.0f}", "MWh/yr",      f"Annual AC Energy ({n_yr:.1f} yr avg)")
    _kpi(c2, f"{sy:,.0f}",       "kWh/kWp/yr",  "Specific Yield")
    _kpi(c3, f"{cf:.1f}%",       "",             "Capacity Factor (AC)")
    _kpi(c4, f"{pr:.1f}%",       "",             "Performance Ratio")
    _kpi(c5, f"{ghi:.0f}",       "kWh/m2/yr",   "Annual GHI")
    _kpi(c6, f"{poa:.0f}",       "kWh/m2/yr",   "Annual POA Irradiation")
    _kpi(c7, f"{cloud:.1f}%",    "daytime avg",  "Cloud Shading")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── charts ─────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_chart_monthly_energy(result_df, p["tz"]), use_container_width=True)
    with col_b:
        st.plotly_chart(_chart_daily_profile(result_df, p["tz"]), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(_chart_irradiance(result_df), use_container_width=True)
    with col_d:
        st.plotly_chart(_chart_cloud_shading(result_df), use_container_width=True)

    st.plotly_chart(_chart_annual_cf(result_df, inv), use_container_width=True)

    # ── annual table ───────────────────────────────────────────────────────────
    st.markdown("#### Annual Energy Summary")
    yr_ghi = (result_df.groupby(result_df.index.year)["ghi_wm2"].sum() / 1000.0).round(0).astype(int)
    yr_poa = (result_df.groupby(result_df.index.year)["poa_wm2"].sum() / 1000.0).round(0).astype(int)
    yr_ac  = (result_df.groupby(result_df.index.year)["ac_power_kw"].sum() / 1000.0).round(1)
    yr_sy  = (result_df.groupby(result_df.index.year)["ac_power_kw"].sum() / p["capacity_kwp"]).round(0).astype(int)
    yr_cf  = (result_df.groupby(result_df.index.year)["ac_power_kw"].sum() / (inv * 8760.0) * 100).round(1)
    day    = result_df[result_df["ghi_clear_wm2"] > 5]
    yr_cloud = day.groupby(day.index.year)["cloud_shading_pct"].mean().round(1)

    tbl = pd.DataFrame({
        "GHI (kWh/m2)":           yr_ghi.values,
        "POA (kWh/m2)":           yr_poa.values,
        "AC Energy (MWh)":        yr_ac.values,
        "Spec. Yield (kWh/kWp)":  yr_sy.values,
        "Cap. Factor (%)":        yr_cf.values,
        "Cloud Shading (%)":      yr_cloud.values,
    }, index=yr_ghi.index.astype(str))
    tbl.index.name = "Year"
    st.dataframe(tbl, use_container_width=True)

    # ── CSV download ────────────────────────────────────────────────────────────
    st.markdown("---")

    dl = result_df.copy()
    dl.index = dl.index.tz_localize(None)
    dl.index.name = "datetime_local"

    hdr = "\n".join([
        "# Solar PV Time Series - NASA POWER x pvlib",
        f"# Site: ({p['lat']:.4f}, {p['lon']:.4f}) Elevation: {p['elevation']} m ASL Timezone: {p['tz']}",
        f"# Period: {p['start_year']}-{p['end_year']} ({n_yr:.1f} yr)",
        f"# System: {p['capacity_kwp']:,.0f} kWp DC Tilt: {p['tilt']} deg Azimuth: {p['azimuth']} deg from N DC:AC: {p['dc_ac_ratio']:.2f}",
        f"# Temp coefficient: {p['temp_coeff']:.2f} pct/degC System losses: {p['system_losses']} pct",
        f"# Annual AC Energy: {aey/1000:,.1f} MWh/yr Specific Yield: {sy:,.0f} kWh/kWp/yr CF: {cf:.1f}% PR: {pr:.1f}% Cloud Shading: {cloud:.1f}%",
        "#",
    ])
    buf = io.StringIO()
    buf.write(hdr + "\n")
    dl.to_csv(buf)
    # utf-8-sig adds BOM so Excel opens special characters correctly
    csv_bytes = buf.getvalue().encode("utf-8-sig")

    fname = (f"solar_ts_{p['start_year']}_{p['end_year']}_"
             f"{p['lat']:.3f}_{p['lon']:.3f}.csv")

    st.download_button(
        label=f"Download Time Series CSV  ({n_yr:.0f} yr - {len(result_df):,} rows - hourly local time)",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "Columns: datetime_local, ghi_wm2, ghi_clear_wm2, cloud_shading_pct, "
        "dni_wm2, dhi_wm2, poa_wm2, temp_air_c, temp_cell_c, wind_speed_ms, dc_power_kw, ac_power_kw"
    )

    # ── methodology note ───────────────────────────────────────────────────────
    with st.expander("Methodology & Assumptions"):
        st.markdown(f"""
**Data source**: NASA POWER (CERES/SRB satellite retrieval) · hourly UTC · ~0.5° grid
**Irradiance transposition**: Hay-Davies model (separates isotropic / circumsolar / horizon diffuse)
**Cell temperature**: Faiman model — T_cell = T_air + G_POA / (U₀ + U₁ × v_wind), where U₀ = 25 W·m⁻²·K⁻¹, U₁ = 6.84 W·m⁻²·K⁻¹·(m/s)⁻¹
**DC power**: P_dc = C_kwp × (G_POA / 1000) × [1 + (γ/100) × (T_cell − 25)]
where γ = {p['temp_coeff']:.2f} %/°C (STC reference temperature = 25 °C)
**Inverter clipping**: AC output capped at {inv:,.0f} kW (DC:AC = {p['dc_ac_ratio']:.2f})
**System losses** ({p['system_losses']}%): applied uniformly to AC output — represents soiling, wiring, mismatch, inverter efficiency variation, and availability losses
**Performance Ratio**: PR = Annual AC Energy / (Annual GHI × System kWp) — excludes temperature effects already captured in DC model
**pvlib version**: {pvlib.__version__}
        """)

else:
    st.info("Configure site and system parameters in the sidebar, then click **☀️ Run Analysis**.")
