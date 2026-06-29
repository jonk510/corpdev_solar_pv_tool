# Solar PV Time Series & Energy Analysis Tool

A free, web-based solar energy analysis tool built in Python/Streamlit. Fetches real satellite-derived solar radiation data from NASA and runs it through a photovoltaic physics model to estimate how much electricity a solar array at any location on Earth would produce. No account, no API key, no cost.

---

## Data Source — NASA POWER

All irradiance and weather data comes from [NASA POWER](https://power.larc.nasa.gov) (Prediction of Worldwide Energy Resources), which provides hourly surface solar radiation derived from **CERES/SRB satellite measurements** blended with MERRA-2 reanalysis meteorology. Coverage is 1990 to near-present at roughly 0.5° grid spacing (~55 km). The tool fetches one year at a time and stitches them together. Timestamps are in Local Solar Time (LST) and are converted to the site's IANA civil timezone automatically.

Variables fetched per hour: GHI, DNI, DHI, clear-sky GHI, 2 m air temperature, 10 m wind speed.

---

## Two Modes

### Single Site
Enter coordinates, a date range, and system parameters. Elevation is auto-fetched from SRTM 30 m (OpenTopoData API) and the local timezone is detected automatically. One click runs the full pipeline and displays results.

### Batch Upload
Upload an Excel file with one row per site.

**Required columns:** `site_name`, `latitude`, `longitude`, `capacity_kwp`

**Optional columns** (fall back to sidebar defaults if omitted):

| Column | Default |
|---|---|
| `tilt_deg` | 0 |
| `azimuth_deg` | 0 (hemisphere-dependent) |
| `dc_ac_ratio` | 1.15 |
| `temp_coeff_pct_c` | −0.40 |
| `system_losses_pct` | 8 |
| `elevation_m_asl` | auto-fetched from SRTM |

The tool processes every site in sequence and produces a combined summary CSV download plus per-site result cards.

---

## Inputs

| Parameter | Typical value | What it controls |
|---|---|---|
| DC Capacity (kWp) | Any | Array nameplate size |
| Tilt (°) | ≈ latitude | Panel angle from horizontal |
| Azimuth (°) | 180 (S hemisphere) | Panel facing direction (0=N, 180=S) |
| DC:AC Ratio | 1.10–1.25 | Inverter sizing relative to array |
| Temp. Coefficient (%/°C) | −0.40 | Power loss above 25°C per degree |
| System Losses (%) | 5–12 | Wiring, soiling, mismatch, inverter, etc. |
| Start/End Year | 1990–present | Period to average over |
| Resolution | Hourly or 30-min | Native or synthetic sub-hourly |

---

## Calculation Pipeline

1. **GHI → POA transposition** using the Hay-Davies model (pvlib `get_total_irradiance`). Accounts for direct beam tilt factor, anisotropic sky diffuse (circumsolar + horizon brightening), and ground reflection (albedo 0.25). Extraterrestrial DNI computed via Spencer's Fourier series.

2. **Cell temperature** via the Faiman (1988) model:
   ```
   T_cell = T_air + G_POA / (U0 + U1 × wind_speed)
   ```
   U0 = 25 W/m²·K, U1 = 6.84 W·s/m³·K (open-rack glass/glass). Higher wind = cooler cells = more power.

3. **DC power** — linear temperature-corrected model:
   ```
   P_dc = kWp × (G_POA / 1000) × [1 + (γ/100) × (T_cell − 25)]
   ```

4. **AC power** — inverter clipping at `kWp / DC:AC ratio`, then uniform system loss derate applied.

5. **Energy summation** — all sums use `power × interval_h` so results are correct for both hourly and 30-min data.

---

## Synthetic 30-min Resolution

When selected, hourly NASA POWER GHI is disaggregated to 30-min using clearness-index interpolation:

1. pvlib Ineichen clear-sky GHI computed at 30-min resolution.
2. Hourly clearness index `Kt = GHI / GHI_clear` interpolated linearly to 30-min.
3. `GHI_30 = Kt_30 × GHI_clear_30`, clipped to ≥ 0.
4. The `:00` mark of each hour is anchored back to the original NASA POWER value to prevent energy drift between the two clear-sky models (NASA POWER CERES vs. pvlib Ineichen).
5. DNI and DHI re-derived at 30-min via Erbs decomposition.

---

## Outputs & KPIs

### KPI Cards

| Metric | Formula | Notes |
|---|---|---|
| Annual AC Energy (MWh/yr) | Sum of P_AC × interval_h | Multi-year average |
| Specific Yield (kWh/kWp/yr) | AEY / C_kWp | Size-independent productivity |
| Capacity Factor (%) | AEY / (P_inverter_kW × 8760) | AC inverter referenced |
| Performance Ratio (%) | AEY / (POA × C_kWp) | POA-referenced per IEC 61724-1 |
| Annual GHI (kWh/m²/yr) | — | Horizontal irradiation |
| Annual POA (kWh/m²/yr) | — | Tilted plane irradiation |
| Cloud Shading (%) | Mean of (1 − GHI/GHI_clear) × 100 | Daytime only (GHI_clear > 5 W/m²) |

### Charts

- Average monthly AC energy output (bar)
- Average diurnal power profile by season (line, 4 seasons)
- Monthly mean irradiance — clear-sky GHI / all-sky GHI / POA (line)
- Monthly average cloud shading % (bar)
- Annual capacity factor by year (bar) — shows inter-annual variability

### Annual Summary Table

Per-year breakdown of GHI, POA, AC energy, specific yield, CF, and cloud shading.

### CSV Download

Full time series with columns: `datetime_local`, `ghi_wm2`, `ghi_clear_wm2`, `cloud_shading_pct`, `dni_wm2`, `dhi_wm2`, `poa_wm2`, `temp_air_c`, `temp_cell_c`, `wind_speed_ms`, `dc_power_kw`, `ac_power_kw`.

UTF-8 with BOM encoding so Excel opens it correctly. Metadata header lines embedded as comments at the top of the file.

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Dependencies

```
streamlit>=1.35
pandas>=2.0
numpy>=1.26
scipy>=1.13
plotly>=5.22
requests>=2.31
pvlib>=0.11
timezonefinder>=6.5
pyproj>=3.6
openpyxl>=3.1
matplotlib>=3.9
folium>=0.17
streamlit-folium>=0.21
```

---

## Limitations

- **~55 km spatial resolution** — local shading, terrain, and microclimate are not captured
- **No module degradation or detailed shading model** — folded into the system losses input
- **NASA POWER accuracy** — ±1–3% globally on average, but can be larger at individual sites in complex terrain or high-aerosol regions
- **30-min output is synthetic** — preserves hourly totals but within-hour distribution is modelled, not measured
- **Simple DC model** — no advanced inverter efficiency curves or spectral correction
- Suitable for **early-stage screening and feasibility studies**, not bankable yield assessment
