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

_BATCH_COLS = {
    "required": ["site_name", "latitude", "longitude", "capacity_kwp"],
    "optional": {
        "tilt_deg":           None,   # auto: round(abs(lat))
        "azimuth_deg":        0,
        "dc_ac_ratio":        1.15,
        "temp_coeff_pct_c":  -0.40,
        "system_losses_pct":  8,
        "elevation_m_asl":    None,   # auto: SRTM lookup
    },
}

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
      NASA POWER (CERES/SRB) &middot; pvlib &middot; Single Site &amp; Batch
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


# ── NASA POWER fetch ──────────────────────────────────────────────────────────
def _fetch_nasa_year(lat: float, lon: float, year: int) -> pd.DataFrame:
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
            f"NASA POWER error {resp.status_code} for {year}: {resp.text[:300]}"
        )
    param_data = resp.json()["properties"]["parameter"]
    first_key  = next(iter(param_data))
    timestamps = list(param_data[first_key].keys())
    df = pd.DataFrame(param_data, index=timestamps)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    # NASA POWER hourly data is in Local Solar Time; localize with longitude-based offset
    lst_tz = datetime.timezone(datetime.timedelta(hours=round(lon / 15.0)))
    df.index = df.index.tz_localize(lst_tz)
    return df


@st.cache_data(show_spinner=False)
def fetch_nasa_power(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
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


# ── 30-min synthetic disaggregation ──────────────────────────────────────────
def disaggregate_30min(df: pd.DataFrame, location: pvlib.location.Location, tz: str) -> pd.DataFrame:
    """
    Synthetic 30-min disaggregation via clearness-index interpolation.
    1. Compute pvlib clear-sky GHI at 30-min using Ineichen model.
    2. Linearly interpolate hourly clearness index (Kt = GHI/GHI_clear) to 30-min.
    3. Reconstruct GHI_30 = Kt_30 * GHI_clear_30.
    4. Decompose to DNI/DHI via Erbs model.
    5. Interpolate temperature and wind linearly.
    """
    # Build 30-min index spanning same period
    idx_30 = pd.date_range(
        df.index[0], df.index[-1], freq="30min", tz=df.index.tz
    )

    # Clear-sky at 30-min (Ineichen)
    times_30_local = idx_30.tz_convert(tz)
    cs_30 = location.get_clearsky(times_30_local)
    ghi_cs_30 = pd.Series(cs_30["ghi"].values, index=idx_30).clip(lower=0)

    # Hourly clearness index Kt (only where clear-sky > 5 W/m²)
    cs_h = pd.Series(df["ghi_clear"].values, index=df.index)
    kt_h = pd.Series(
        np.where(cs_h > 5, (df["ghi"].values / cs_h.values).clip(0, 1), 0.0),
        index=df.index,
    )

    # Interpolate Kt to 30-min
    kt_30 = kt_h.reindex(kt_h.index.union(idx_30)).interpolate("linear").reindex(idx_30).fillna(0)

    # Reconstruct GHI at :30 marks via Kt × Ineichen clear-sky
    ghi_30_s = pd.Series((kt_30.values * ghi_cs_30.values).clip(0), index=idx_30)
    # Anchor :00 marks to original NASA POWER values to preserve energy conservation
    # (avoids systematic bias from CERES vs Ineichen clear-sky model mismatch)
    hourly_in_30 = df.index.intersection(idx_30)
    ghi_30_s[hourly_in_30] = df.loc[hourly_in_30, "ghi"].values
    ghi_30 = ghi_30_s.values

    # Decompose GHI → DNI + DHI via Erbs
    solpos_30 = location.get_solarposition(times_30_local)
    erbs = pvlib.irradiance.erbs(ghi_30, solpos_30["zenith"].values, idx_30)
    dni_30 = np.clip(erbs["dni"].values, 0, None)
    dhi_30 = np.clip(erbs["dhi"].values, 0, None)

    # Interpolate met data
    def _interp(series):
        return series.reindex(series.index.union(idx_30)).interpolate("linear").reindex(idx_30).bfill()

    temp_30  = _interp(pd.Series(df["temp_air"].values,   index=df.index))
    wind_30  = _interp(pd.Series(df["wind_speed"].values, index=df.index))

    return pd.DataFrame({
        "ghi":        ghi_30,
        "dni":        dni_30,
        "dhi":        dhi_30,
        "ghi_clear":  ghi_cs_30.values,
        "temp_air":   temp_30.values,
        "wind_speed": wind_30.values,
    }, index=idx_30)


# ── pvlib pipeline ────────────────────────────────────────────────────────────
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
    resolution: str = "Hourly",
) -> tuple:
    """pvlib pipeline → (result_df, meta_dict). resolution: 'Hourly' or '30-min'."""
    location = pvlib.location.Location(lat, lon, tz=tz, altitude=elevation)

    if resolution == "30-min":
        df = disaggregate_30min(df, location, tz)

    interval_h = (df.index[1] - df.index[0]).total_seconds() / 3600.0

    times_local = df.index.tz_convert(tz)
    solpos = location.get_solarposition(times_local)
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)

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

    temp_cell_arr = pvlib.temperature.faiman(
        poa_global=poa.values,
        temp_air=df["temp_air"].values,
        wind_speed=df["wind_speed"].values,
    )
    temp_cell = pd.Series(temp_cell_arr, index=df.index)

    dc_power    = capacity_kwp * (poa / 1000.0) * (1.0 + (temp_coeff / 100.0) * (temp_cell - 25.0))
    dc_power    = dc_power.clip(lower=0)
    inverter_kw = capacity_kwp / dc_ac_ratio
    ac_power    = (dc_power.clip(upper=inverter_kw) * (1.0 - system_losses_pct / 100.0)).clip(lower=0)

    ghi_clear = df["ghi_clear"]
    cloud_t   = np.where(ghi_clear > 5, df["ghi"] / ghi_clear, np.nan)
    cloud_t   = np.clip(cloud_t, 0.0, 1.0)
    cloud_pct = pd.Series((1.0 - cloud_t) * 100, index=df.index)

    result = pd.DataFrame({
        "ghi_wm2":           df["ghi"].round(1),
        "ghi_clear_wm2":     ghi_clear.round(1),
        "cloud_shading_pct": cloud_pct.round(1),
        "dni_wm2":           df["dni"].round(1),
        "dhi_wm2":           df["dhi"].round(1),
        "poa_wm2":           poa.round(1),
        "temp_air_c":        df["temp_air"].round(2),
        "temp_cell_c":       temp_cell.round(2),
        "wind_speed_ms":     df["wind_speed"].round(2),
        "dc_power_kw":       dc_power.round(4),
        "ac_power_kw":       ac_power.round(4),
    }, index=df.index)

    n_years = (df.index[-1] - df.index[0]).days / 365.25
    ann_ghi  = df["ghi"].sum()  * interval_h / 1000.0 / n_years   # kWh/m2/yr
    ann_poa  = poa.sum()        * interval_h / 1000.0 / n_years   # kWh/m2/yr
    ann_ac   = ac_power.sum()   * interval_h           / n_years   # kWh/yr
    sy       = ann_ac / capacity_kwp
    cf       = ann_ac / (inverter_kw * 8760.0)
    pr       = ann_ac / (ann_poa * capacity_kwp)   # IEC 61724-1: PR referenced to POA, not GHI
    mean_cloud = (1.0 - float(np.nanmean(cloud_t))) * 100

    meta = {
        "n_years": n_years,
        "interval_hours":         interval_h,
        "annual_ghi_kwh_m2":      ann_ghi,
        "annual_poa_kwh_m2":      ann_poa,
        "annual_ac_kwh":          ann_ac,
        "specific_yield":         sy,
        "capacity_factor":        cf,
        "performance_ratio":      pr,
        "inverter_kw":            inverter_kw,
        "mean_cloud_shading_pct": mean_cloud,
    }
    return result, meta


# ── chart helpers ─────────────────────────────────────────────────────────────
_NAVY  = "#1B3A6B"
_TEAL  = "#0F766E"
_AMBER = "#D97706"
_SLATE = "#64748B"


def _chart_monthly_energy(result_df: pd.DataFrame, tz: str, interval_h: float) -> go.Figure:
    local = result_df["ac_power_kw"].copy()
    local.index = local.index.tz_convert(tz)
    # Sum over each month then average across years; multiply by interval_h for kWh→MWh
    avg = (local.groupby([local.index.year, local.index.month]).sum()
               .groupby(level=1).mean() * interval_h / 1000.0)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Bar(
        x=months, y=avg.values.round(1),
        marker_color=_NAVY,
        text=[f"{v:.0f}" for v in avg.values], textposition="outside",
    ))
    fig.update_layout(title="Average Monthly AC Energy Output", yaxis_title="MWh", height=310,
                      margin=dict(l=50,r=20,t=40,b=40),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def _chart_daily_profile(result_df: pd.DataFrame, tz: str) -> go.Figure:
    local = result_df["ac_power_kw"].copy()
    local.index = local.index.tz_convert(tz)
    decimal_hour = local.index.hour + local.index.minute / 60.0
    df2 = pd.DataFrame({"kw": local.values, "hour": decimal_hour, "month": local.index.month})
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
        xaxis=dict(title="Hour (local time)", dtick=2, range=[0, 24]),
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=grp["ghi_clear_wm2"].mean().round(1).values,
                             name="Clear-sky GHI", line=dict(color="#CBD5E1", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=months, y=grp["ghi_wm2"].mean().round(1).values,
                             name="GHI (all-sky)", line=dict(color=_SLATE, width=2)))
    fig.add_trace(go.Scatter(x=months, y=grp["poa_wm2"].mean().round(1).values,
                             name="POA (tilted)", line=dict(color=_AMBER, width=2)))
    fig.update_layout(title="Monthly Mean Irradiance — Clear-sky / GHI / POA",
                      yaxis_title="W/m2", height=310,
                      margin=dict(l=50,r=20,t=40,b=40),
                      plot_bgcolor="white", paper_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def _chart_cloud_shading(result_df: pd.DataFrame) -> go.Figure:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    day = result_df[result_df["ghi_clear_wm2"] > 5]
    if len(day) == 0:
        return go.Figure().update_layout(title="Cloud Shading (no daytime data)", height=310)
    # reindex to 1-12 so month labels always align correctly even if some months have no data
    shading_m = (day.groupby(day.index.month)["cloud_shading_pct"]
                    .mean()
                    .reindex(range(1, 13))
                    .round(1))
    y_max = float(np.nanmax(shading_m.values)) if not np.all(np.isnan(shading_m.values)) else 50
    fig = go.Figure(go.Bar(
        x=months, y=shading_m.values, marker_color=_SLATE,
        text=[f"{v:.0f}%" if not np.isnan(v) else "" for v in shading_m.values],
        textposition="outside",
    ))
    fig.update_layout(title="Monthly Average Cloud Shading (% of Clear-Sky GHI Lost)",
                      yaxis_title="Cloud Shading (%)", height=310,
                      margin=dict(l=50,r=20,t=40,b=40),
                      plot_bgcolor="white", paper_bgcolor="white",
                      yaxis_range=[0, y_max * 1.25])
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def _chart_annual_cf(result_df: pd.DataFrame, inverter_kw: float, interval_h: float) -> go.Figure:
    ann = result_df.groupby(result_df.index.year)["ac_power_kw"].sum() * interval_h
    cf  = (ann / (inverter_kw * 8760.0) * 100).round(1)
    fig = go.Figure(go.Bar(
        x=cf.index.astype(str), y=cf.values, marker_color=_TEAL,
        text=[f"{v:.1f}%" for v in cf.values], textposition="outside",
    ))
    fig.update_layout(title="Annual AC Capacity Factor by Year",
                      yaxis_title="Capacity Factor (%)", height=280,
                      margin=dict(l=50,r=20,t=40,b=40),
                      plot_bgcolor="white", paper_bgcolor="white",
                      yaxis_range=[0, max(cf.values) * 1.25])
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# ── results renderer (shared by single-site and batch per-site) ───────────────
def _kpi(col, val, unit, label):
    col.markdown(
        f'<div class="kpi-card"><div class="val">{val}</div>'
        f'<div class="unit">{unit}</div><div class="lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )


def _render_results(result_df, meta, p):
    n_yr   = meta["n_years"]
    ih     = meta["interval_hours"]
    aey    = meta["annual_ac_kwh"]
    sy     = meta["specific_yield"]
    cf     = meta["capacity_factor"] * 100
    pr     = meta["performance_ratio"] * 100
    ghi    = meta["annual_ghi_kwh_m2"]
    poa    = meta["annual_poa_kwh_m2"]
    inv    = meta["inverter_kw"]
    cloud  = meta["mean_cloud_shading_pct"]
    cap    = p["capacity_kwp"]

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    _kpi(c1, f"{aey/1000:,.0f}", "MWh/yr",     f"Annual AC Energy ({n_yr:.1f} yr avg)")
    _kpi(c2, f"{sy:,.0f}",       "kWh/kWp/yr", "Specific Yield")
    _kpi(c3, f"{cf:.1f}%",       "",            "Capacity Factor (AC)")
    _kpi(c4, f"{pr:.1f}%",       "",            "Performance Ratio")
    _kpi(c5, f"{ghi:.0f}",       "kWh/m2/yr",  "Annual GHI")
    _kpi(c6, f"{poa:.0f}",       "kWh/m2/yr",  "Annual POA Irradiation")
    _kpi(c7, f"{cloud:.1f}%",    "daytime avg", "Cloud Shading")
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_chart_monthly_energy(result_df, p["tz"], ih), use_container_width=True)
    with col_b:
        st.plotly_chart(_chart_daily_profile(result_df, p["tz"]), use_container_width=True)
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(_chart_irradiance(result_df), use_container_width=True)
    with col_d:
        st.plotly_chart(_chart_cloud_shading(result_df), use_container_width=True)
    st.plotly_chart(_chart_annual_cf(result_df, inv, ih), use_container_width=True)

    st.markdown("#### Annual Energy Summary")
    yr_ghi   = (result_df.groupby(result_df.index.year)["ghi_wm2"].sum()  * ih / 1000.0).round(0).astype(int)
    yr_poa   = (result_df.groupby(result_df.index.year)["poa_wm2"].sum()  * ih / 1000.0).round(0).astype(int)
    yr_ac    = (result_df.groupby(result_df.index.year)["ac_power_kw"].sum() * ih / 1000.0).round(1)
    yr_sy    = (result_df.groupby(result_df.index.year)["ac_power_kw"].sum() * ih / cap).round(0).astype(int)
    yr_cf    = (result_df.groupby(result_df.index.year)["ac_power_kw"].sum() * ih / (inv * 8760.0) * 100).round(1)
    day      = result_df[result_df["ghi_clear_wm2"] > 5]
    yr_cloud = day.groupby(day.index.year)["cloud_shading_pct"].mean().round(1)
    tbl = pd.DataFrame({
        "GHI (kWh/m2)":          yr_ghi.values,
        "POA (kWh/m2)":          yr_poa.values,
        "AC Energy (MWh)":       yr_ac.values,
        "Spec. Yield (kWh/kWp)": yr_sy.values,
        "Cap. Factor (%)":       yr_cf.values,
        "Cloud Shading (%)":     yr_cloud.values,
    }, index=yr_ghi.index.astype(str))
    tbl.index.name = "Year"
    st.dataframe(tbl, use_container_width=True)

    st.markdown("---")
    res_label = "30min" if ih < 1.0 else "hourly"
    dl = result_df.copy()
    dl.index = dl.index.tz_localize(None)
    dl.index.name = "datetime_local"
    hdr = "\n".join([
        "# Solar PV Time Series - NASA POWER x pvlib",
        f"# Site: ({p['lat']:.4f}, {p['lon']:.4f}) Elev: {p['elevation']} m ASL TZ: {p['tz']}",
        f"# Period: {p['start_year']}-{p['end_year']} ({n_yr:.1f} yr) Resolution: {res_label}",
        f"# System: {cap:,.0f} kWp DC Tilt: {p['tilt']} deg Az: {p['azimuth']} deg DC:AC: {p['dc_ac_ratio']:.2f}",
        f"# Temp coeff: {p['temp_coeff']:.2f} pct/degC System losses: {p['system_losses']} pct",
        f"# AEY: {aey/1000:,.1f} MWh/yr SY: {sy:,.0f} kWh/kWp/yr CF: {cf:.1f}% PR: {pr:.1f}% Cloud: {cloud:.1f}%",
        "#",
    ])
    buf = io.StringIO()
    buf.write(hdr + "\n")
    dl.to_csv(buf)
    csv_bytes = buf.getvalue().encode("utf-8-sig")
    fname = f"solar_ts_{p['start_year']}_{p['end_year']}_{p['lat']:.3f}_{p['lon']:.3f}_{res_label}.csv"
    st.download_button(
        label=f"Download Time Series CSV  ({n_yr:.0f} yr - {len(result_df):,} rows - {res_label} local time)",
        data=csv_bytes, file_name=fname, mime="text/csv", use_container_width=True,
    )
    st.caption(
        "Columns: datetime_local, ghi_wm2, ghi_clear_wm2, cloud_shading_pct, "
        "dni_wm2, dhi_wm2, poa_wm2, temp_air_c, temp_cell_c, wind_speed_ms, dc_power_kw, ac_power_kw"
    )


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    app_mode = st.radio("Mode", ["Single Site", "Batch Upload"], horizontal=True)
    st.markdown("---")

    if app_mode == "Single Site":
        st.markdown("### Location")
        lat = st.number_input("Latitude",  value=-31.9505, min_value=-90.0,  max_value=90.0,  step=0.001, format="%.4f")
        lon = st.number_input("Longitude", value=115.8605, min_value=-180.0, max_value=180.0, step=0.001, format="%.4f")
        elev_auto = _get_elevation(lat, lon)
        elevation = st.number_input(
            "Elevation (m ASL)", value=elev_auto, min_value=0, max_value=9000, step=1,
            key=f"elev_{lat:.4f}_{lon:.4f}",
            help="Auto-fetched from SRTM 30m. Override if needed.",
        )

    st.markdown("### Date Range")
    st.caption(f"NASA POWER: 1990 - {_MAX_YEAR}")
    start_year = st.number_input("Start Year", value=_START_YEAR_DEFAULT, min_value=_MIN_YEAR, max_value=_MAX_YEAR, step=1)
    end_year   = st.number_input("End Year",   value=_END_YEAR_DEFAULT,   min_value=_MIN_YEAR, max_value=_MAX_YEAR, step=1)
    if end_year < start_year:
        st.warning("End year must be >= start year.")
        end_year = start_year

    st.markdown("### Time Resolution")
    resolution = st.radio(
        "Output resolution", ["Hourly", "30-min (synthetic)"],
        help="30-min uses clearness-index interpolation + Erbs decomposition. "
             "Improves inverter clipping accuracy; does not add within-hour cloud variability.",
    )

    if app_mode == "Single Site":
        st.markdown("### System Parameters")
        capacity_kwp = st.number_input(
            "Capacity (kWp DC)", value=10_000.0, min_value=1.0, max_value=2_000_000.0, step=500.0,
        )
        tilt = st.slider("Tilt (deg from horizontal)", 0, 90, min(int(round(abs(lat))), 60))
        azimuth = st.slider(
            "Azimuth (deg from North)", 0, 359, 0,
            help="0 = North (S. hemisphere optimal). 180 = South (N. hemisphere optimal).",
        )
        dc_ac_ratio = st.slider("DC:AC Ratio", 1.0, 1.5, 1.15, 0.05)
        temp_coeff  = st.slider("Temp Coefficient (%/degC)", -0.60, -0.20, -0.40, 0.01)
        system_losses = st.slider("System Losses (%)", 0, 30, 8)

    st.markdown("---")
    if app_mode == "Single Site":
        run_btn = st.button("Run Analysis", use_container_width=True)
    else:
        st.markdown("**Excel columns required:**")
        st.caption("Required: site_name, latitude, longitude, capacity_kwp")
        st.caption("Optional: tilt_deg, azimuth_deg, dc_ac_ratio, temp_coeff_pct_c, system_losses_pct, elevation_m_asl")
        uploaded_file = st.file_uploader("Upload sites Excel (.xlsx)", type=["xlsx"])
        run_btn = st.button("Run Batch Analysis", use_container_width=True, disabled=uploaded_file is None)


# ── single site mode ──────────────────────────────────────────────────────────
if app_mode == "Single Site":
    col_map, col_info = st.columns([3, 2])
    with col_map:
        m = folium.Map(location=[lat, lon], zoom_start=8, tiles="CartoDB positron")
        folium.Marker([lat, lon], tooltip=f"{lat:.4f}, {lon:.4f}",
                      icon=folium.Icon(color="orange", icon="sun-o", prefix="fa")).add_to(m)
        st_folium(m, height=260, width=None, returned_objects=[])
    with col_info:
        tz_auto = _get_tz(lat, lon)
        st.markdown(f"""
**Site**  Lat/Lon: `{lat:.4f} / {lon:.4f}`  Elev: `{elevation} m`  TZ: `{tz_auto}`
**System**  `{capacity_kwp:,.0f} kWp`  Tilt: `{tilt}deg`  Az: `{azimuth}deg`  DC:AC: `{dc_ac_ratio:.2f}`
**Data**  NASA POWER  `{start_year}-{end_year}`  Resolution: `{resolution}`
        """)
    st.markdown("---")

    if run_btn:
        tz = _get_tz(lat, lon)
        n_req = end_year - start_year + 1
        with st.spinner(f"Fetching NASA POWER data ({start_year}-{end_year}, {n_req} calls)..."):
            try:
                df_raw = fetch_nasa_power(lat, lon, start_year, end_year)
            except Exception as e:
                st.error(f"NASA POWER fetch failed: {e}")
                st.stop()
        label = "30-min" if "30" in resolution else "hourly"
        with st.spinner(f"Running pvlib pipeline ({len(df_raw):,} hourly records -> {label})..."):
            try:
                result_df, meta = run_solar_pipeline(
                    df_raw, lat, lon, elevation, tz,
                    tilt, azimuth, capacity_kwp, dc_ac_ratio, temp_coeff, system_losses,
                    resolution="30-min" if "30" in resolution else "Hourly",
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()
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

    if "solar_result" in st.session_state:
        _render_results(
            st.session_state["solar_result"],
            st.session_state["solar_meta"],
            st.session_state["solar_params"],
        )
    else:
        st.info("Configure parameters in the sidebar, then click **Run Analysis**.")


# ── batch mode ────────────────────────────────────────────────────────────────
else:
    st.markdown("### Batch Site Analysis")

    with st.expander("Excel template — required and optional columns"):
        st.markdown("""
| Column | Required | Default | Notes |
|---|---|---|---|
| `site_name` | Yes | — | Label for site |
| `latitude` | Yes | — | Decimal degrees |
| `longitude` | Yes | — | Decimal degrees |
| `capacity_kwp` | Yes | — | DC array capacity (kWp) |
| `tilt_deg` | No | `round(abs(lat))` | Panel tilt from horizontal |
| `azimuth_deg` | No | `0` | Degrees from North (0=N optimal S. hemisphere, 180=S optimal N. hemisphere) |
| `dc_ac_ratio` | No | `1.15` | DC capacity / inverter rating |
| `temp_coeff_pct_c` | No | `-0.40` | Power loss per degC above 25 |
| `system_losses_pct` | No | `8` | Soiling, wiring, availability etc. |
| `elevation_m_asl` | No | Auto (SRTM) | Leave blank to auto-fetch |
        """)

    if uploaded_file is None:
        st.info("Upload an Excel file using the sidebar to begin batch processing.")

    elif run_btn:
        # Parse Excel
        try:
            sites_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            st.stop()

        missing = [c for c in _BATCH_COLS["required"] if c not in sites_df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        n_sites = len(sites_df)
        res_key = "30-min" if "30" in resolution else "Hourly"
        progress = st.progress(0, text="Starting batch...")
        summary_rows = []
        batch_errors = []

        for i, row in sites_df.iterrows():
            site  = str(row["site_name"])
            lat_i = float(row["latitude"])
            lon_i = float(row["longitude"])
            cap_i = float(row["capacity_kwp"])

            # Optional params with defaults
            def _opt(col, default):
                v = row.get(col, default)
                if pd.isna(v) if isinstance(v, float) else (v is None):
                    return default
                return v

            tilt_i = _opt("tilt_deg",          min(int(round(abs(lat_i))), 60))
            az_i   = _opt("azimuth_deg",        0)
            dc_i   = _opt("dc_ac_ratio",        1.15)
            tc_i   = _opt("temp_coeff_pct_c",  -0.40)
            sl_i   = _opt("system_losses_pct",  8)
            elev_v = row.get("elevation_m_asl", None)
            elev_i = (int(round(float(elev_v)))
                      if (elev_v is not None and not (isinstance(elev_v, float) and pd.isna(elev_v)))
                      else _get_elevation(lat_i, lon_i))

            progress.progress((i) / n_sites, text=f"Processing {site} ({i+1}/{n_sites})...")
            try:
                tz_i = _get_tz(lat_i, lon_i)
                df_raw_i = fetch_nasa_power(lat_i, lon_i, start_year, end_year)
                _, meta_i = run_solar_pipeline(
                    df_raw_i, lat_i, lon_i, elev_i, tz_i,
                    float(tilt_i), float(az_i), cap_i, float(dc_i), float(tc_i), float(sl_i),
                    resolution=res_key,
                )
                summary_rows.append({
                    "site_name":              site,
                    "latitude":               lat_i,
                    "longitude":              lon_i,
                    "elevation_m_asl":        elev_i,
                    "capacity_kwp":           cap_i,
                    "tilt_deg":               int(tilt_i),
                    "azimuth_deg":            int(az_i),
                    "dc_ac_ratio":            round(float(dc_i), 2),
                    "annual_ghi_kwh_m2":      round(meta_i["annual_ghi_kwh_m2"]),
                    "annual_poa_kwh_m2":      round(meta_i["annual_poa_kwh_m2"]),
                    "annual_ac_mwh_yr":       round(meta_i["annual_ac_kwh"] / 1000.0, 1),
                    "specific_yield_kwh_kwp": round(meta_i["specific_yield"]),
                    "capacity_factor_pct":    round(meta_i["capacity_factor"] * 100, 1),
                    "performance_ratio_pct":  round(meta_i["performance_ratio"] * 100, 1),
                    "cloud_shading_pct":      round(meta_i["mean_cloud_shading_pct"], 1),
                })
            except Exception as e:
                batch_errors.append(f"{site}: {e}")

        progress.progress(1.0, text="Done.")

        if batch_errors:
            st.warning("Some sites failed:\n" + "\n".join(batch_errors))

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.success(f"Processed {len(summary_rows)} of {n_sites} sites.")
            st.markdown("#### Results Summary")
            st.dataframe(summary_df.set_index("site_name"), use_container_width=True)

            # Download summary CSV
            buf = io.StringIO()
            buf.write(f"# Solar PV Batch Analysis - NASA POWER x pvlib\n")
            buf.write(f"# Period: {start_year}-{end_year}  Resolution: {res_key}\n#\n")
            summary_df.to_csv(buf, index=False)
            st.download_button(
                label=f"Download Batch Summary CSV  ({len(summary_rows)} sites)",
                data=buf.getvalue().encode("utf-8-sig"),
                file_name=f"solar_batch_{start_year}_{end_year}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    elif uploaded_file is not None:
        # File uploaded but button not clicked yet — show preview
        try:
            preview = pd.read_excel(uploaded_file)
            st.markdown(f"**{len(preview)} sites loaded** — click **Run Batch Analysis** in the sidebar to proceed.")
            st.dataframe(preview, use_container_width=True)
        except Exception as e:
            st.error(f"Could not preview file: {e}")
