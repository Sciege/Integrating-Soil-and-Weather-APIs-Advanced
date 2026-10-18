"""ITD112 Laboratory Exercise 1: Integrating Soil and Weather APIs.

Dark Tech Console implementation following antislop principles:
  1 Configuration: monitoring sites, endpoints, properties, and luminous dark palette
  2 Request Builder: percent-encoded URL query generation
  3 Defensive Transmission: connection pooling, retries, disk cache, and fair use delays
  4 Schema Parsing: tabular flattening and unit conversions via d_factor
  5 Relational Integration: broadcast join and derived water balance
  6 Analytical Visualizations: Plotly dark figures for questions Q1 through Q6
  7 Interface Layout: dark metrics, compact toolbar, and paired dashboard charts
  8 Main Application: pipeline execution and export

Sources: SoilGrids 2.0 (ISRIC, CC BY 4.0), Open-Meteo Historical Weather API (CC BY 4.0).
"""

import hashlib
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1 Configuration: monitoring sites, endpoints, properties, and palettes
SITES = pd.DataFrame(
    [
        ("Iligan City", 8.200, 124.300),
        ("Malaybalay", 8.150, 125.130),
        ("Valencia", 7.900, 125.090),
        ("Davao (Calinan)", 7.190, 125.460),
        ("General Santos", 6.150, 125.150),
    ],
    columns=["site", "lat", "lon"],
)

SOIL_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

SOIL_PROPERTIES = ["clay", "sand", "silt", "phh2o", "soc"]
SOIL_DEPTH = "0-5cm"
DAILY_VARS = ["temperature_2m_mean", "precipitation_sum", "et0_fao_evapotranspiration"]
TIMEZONE = "Asia/Manila"

CACHE_DIR = Path(__file__).parent / "cache"
SOILGRIDS_PAUSE_S = 12  # Fair use delay: ~5 SoilGrids calls per minute

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# High-contrast dark console palette with WCAG AA compliance
COLORS = {
    "canvas": "#0B0F12",
    "surface": "#151C22",
    "border": "#27333F",
    "grid": "#1E293B",
    "text": "#F1F5F9",
    "text_muted": "#94A3B8",
    "teal": "#14B8A6",
    "teal_soft": "#134E4A",
    "sky": "#38BDF8",
    "sky_pale": "#7DD3FC",
    "sun": "#FBBF24",
    "clay": "#FB923C",
    "sand": "#FDE047",
    "silt": "#4ADE80",
}

SOURCE_NOTE = (
    f"Sources: SoilGrids 2.0 (ISRIC, CC BY 4.0), {SOIL_DEPTH} depth mean; "
    "Open-Meteo Historical Weather API (CC BY 4.0). "
    "SoilGrids values represent spatial model predictions with quantified uncertainty, "
    "not physical field sensor measurements at the coordinate."
)


# 2 Build the request
def full_url(url, params):
    """Return prepared URL with percent-encoded query parameters."""
    return requests.Request("GET", url, params=params).prepare().url


def build_weather_request(sites, start, end):
    """Open-Meteo accepts comma-separated coordinates, so one request covers every site."""
    params = {
        "latitude": ",".join(str(v) for v in sites["lat"]),
        "longitude": ",".join(str(v) for v in sites["lon"]),
        "start_date": str(start),
        "end_date": str(end),
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    }
    return {
        "api": "Open-Meteo",
        "label": "All selected sites",
        "url": WEATHER_URL,
        "params": params,
        "full_url": full_url(WEATHER_URL, params),
    }


def build_soil_request(site, lat, lon):
    """SoilGrids takes one point per call. Property key repeats once per soil property."""
    params = [("lat", lat), ("lon", lon), ("depth", SOIL_DEPTH), ("value", "mean")]
    params += [("property", p) for p in SOIL_PROPERTIES]
    return {
        "api": "SoilGrids",
        "label": site,
        "url": SOIL_URL,
        "params": params,
        "full_url": full_url(SOIL_URL, params),
    }


# 3 Send the request
@st.cache_resource
def http_session():
    """Pool HTTP connections and retry on rate limits (429) or transient server errors (5xx)."""
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "itd112-lab1/1.0"
    return session


def send(req):
    """Execute GET request with on-disk caching to prevent redundant external calls."""
    key = hashlib.sha1(
        (req["url"] + json.dumps(req["params"], sort_keys=True, default=str)).encode()
    ).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return {
            "status": None,
            "from_cache": True,
            "elapsed_s": None,
            "bytes": len(text),
            "data": json.loads(text),
        }

    t0 = time.perf_counter()
    response = http_session().get(req["url"], params=req["params"], timeout=60)
    elapsed = time.perf_counter() - t0
    response.raise_for_status()

    data = response.json()
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return {
        "status": response.status_code,
        "from_cache": False,
        "elapsed_s": round(elapsed, 2),
        "bytes": len(response.content),
        "data": data,
    }


# 4 Parse the JSON
def parse_weather(data, sites):
    """Turn daily parallel arrays into tabular rows with site identifiers."""
    results = data if isinstance(data, list) else [data]
    frames = []
    for site, res in zip(sites["site"], results):
        df = pd.DataFrame(res["daily"])
        df.insert(0, "site", site)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def parse_soil(data, site):
    """Extract soil layer means and divide by d_factor for conventional units."""
    rows = []
    for layer in data["properties"]["layers"]:
        raw = layer["depths"][0]["values"]["mean"]
        unit = layer["unit_measure"]
        rows.append({
            "site": site,
            "property": layer["name"],
            "raw_mean": raw,
            "mapped_units": unit.get("mapped_units"),
            "d_factor": unit["d_factor"],
            "value": np.nan if raw is None else raw / unit["d_factor"],
            "target_units": unit.get("target_units"),
        })
    return pd.DataFrame(rows)


def soil_table(soil_long, sites):
    """Pivot long soil records into one row per site with latitude and longitude."""
    wide = (
        soil_long.pivot(index="site", columns="property", values="value")
        .reindex(columns=SOIL_PROPERTIES)
        .reset_index()
    )
    wide.columns.name = None
    return sites.merge(wide, on="site", how="left")


# 5 Integrate
def integrate(weather, soil):
    """Left-join static soil properties onto daily weather records and compute water balance."""
    daily = weather.merge(soil, on="site", how="left")
    daily["time"] = pd.to_datetime(daily["time"])
    daily["month"] = daily["time"].dt.month
    daily["water_balance_mm"] = (
        daily["precipitation_sum"] - daily["et0_fao_evapotranspiration"]
    )
    summary = (
        daily.groupby("site", sort=False)
        .agg(
            rain_mm=("precipitation_sum", "sum"),
            et0_mm=("et0_fao_evapotranspiration", "sum"),
            water_balance_mm=("water_balance_mm", "sum"),
            temp_mean_c=("temperature_2m_mean", "mean"),
        )
        .reset_index()
        .merge(soil, on="site")
    )
    return daily, summary


# Dark Console chart formatter
def format_chart(fig, height=380):
    """Apply high-contrast dark console styling."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12, color=COLORS["text"]),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(21, 28, 34, 0.9)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(color=COLORS["text"]),
        ),
        hoverlabel=dict(
            bgcolor="#151C22",
            font_size=12,
            font_color=COLORS["text"],
            bordercolor=COLORS["border"],
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_muted"]),
        title_font=dict(color=COLORS["text"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_muted"]),
        title_font=dict(color=COLORS["text"]),
    )
    return fig


# 6 Charts Q1-Q6
def chart_q1(daily):
    """Q1: Daily mean temperature over time with 7-day rolling smoothing."""
    df = daily.sort_values(["site", "time"]).copy()
    df["temp_7d"] = (
        df.groupby("site")["temperature_2m_mean"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    fig = px.line(
        df,
        x="time",
        y="temp_7d",
        color="site",
        title="Q1: Daily mean temperature (7-day rolling mean)",
        labels={"time": "Date", "temp_7d": "Temperature (°C, 7-day mean)", "site": "Site"},
        color_discrete_sequence=[COLORS["teal"], COLORS["sky"], COLORS["clay"], COLORS["silt"], COLORS["sun"]],
    )
    return format_chart(fig, height=360)


def chart_q2(daily, site):
    """Q2: Monthly precipitation bars versus reference ET0 on a shared mm axis."""
    monthly = (
        daily[daily["site"] == site]
        .groupby("month")[["precipitation_sum", "et0_fao_evapotranspiration"]]
        .sum()
        .reindex(range(1, 13), fill_value=0)
    )
    deficit = monthly["precipitation_sum"] < monthly["et0_fao_evapotranspiration"]
    fig = go.Figure()
    fig.add_bar(
        x=MONTHS,
        y=monthly["precipitation_sum"],
        name="Rainfall (surplus)",
        marker_color=np.where(deficit, COLORS["sky_pale"], COLORS["sky"]),
    )
    fig.add_scatter(
        x=MONTHS,
        y=monthly["et0_fao_evapotranspiration"],
        name="Reference ET0",
        mode="lines+markers",
        line=dict(color=COLORS["sun"], width=2.5),
        marker=dict(size=5),
    )
    fig.update_layout(
        title=f"Q2: Monthly rainfall vs reference ET0: {site}",
        xaxis_title="Month",
    )
    fig.update_yaxes(title="mm per month", rangemode="tozero")
    return format_chart(fig, height=360)


def chart_q3(daily):
    """Q3: Distribution of daily rainfall by month."""
    df = daily.copy()
    df["month_name"] = pd.Categorical(
        df["month"].map(dict(enumerate(MONTHS, start=1))),
        categories=MONTHS,
        ordered=True,
    )
    fig = px.box(
        df,
        x="month_name",
        y="precipitation_sum",
        color="month_name",
        title="Q3: Distribution of daily rainfall by month",
        labels={"month_name": "Month", "precipitation_sum": "Daily rainfall (mm)"},
        color_discrete_sequence=[COLORS["teal"]] * 12,
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(rangemode="tozero")
    return format_chart(fig, height=360)


def chart_q4(summary):
    """Q4: Topsoil sand, silt, and clay normalised to 100 percent."""
    tex = summary.set_index("site")[["sand", "silt", "clay"]].dropna()
    tex = tex.div(tex.sum(axis=1), axis=0).mul(100).sort_values("clay")
    fig = go.Figure()
    for col, color in [("sand", COLORS["sand"]), ("silt", COLORS["silt"]), ("clay", COLORS["clay"])]:
        fig.add_bar(x=tex.index, y=tex[col], name=col.capitalize(), marker_color=color)
    fig.update_layout(
        barmode="stack",
        barnorm="percent",
        title=f"Q4: Topsoil texture ({SOIL_DEPTH}, normalised to 100%)",
        xaxis_title="Site",
        yaxis_title="% of sand + silt + clay",
    )
    return format_chart(fig, height=360)


def chart_q5(summary):
    """Q5: Scatter comparing soil clay percentage and total annual water balance."""
    s = summary.dropna(subset=["clay", "soc"]).copy()
    fig = go.Figure()
    if s.empty:
        fig.update_layout(title="Q5: No soil data available for scatter")
        return format_chart(fig, height=360)

    sizes = 14 + 24 * s["soc"] / s["soc"].max()
    fig.add_scatter(
        x=s["clay"],
        y=s["water_balance_mm"],
        mode="markers+text",
        text=s["site"],
        textposition="top center",
        marker=dict(
            size=sizes,
            color=COLORS["teal"],
            opacity=0.9,
            line=dict(width=1, color=COLORS["canvas"]),
        ),
        name="Sites (size = SOC)",
    )
    if len(s) >= 5:
        slope, intercept = np.polyfit(s["clay"], s["water_balance_mm"], 1)
        r = np.corrcoef(s["clay"], s["water_balance_mm"])[0, 1]
        xs = np.linspace(s["clay"].min(), s["clay"].max(), 50)
        fig.add_scatter(
            x=xs,
            y=slope * xs + intercept,
            mode="lines",
            name=f"Linear fit (r = {r:.2f}, n = {len(s)})",
            line=dict(dash="dash", color=COLORS["sun"], width=2),
        )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=COLORS["text_muted"],
        annotation_text="Rainfall = ET0",
        annotation_position="bottom right",
        annotation_font_color=COLORS["text_muted"],
    )
    fig.update_layout(
        title="Q5: Clay (%) vs annual water balance (Rain - ET0)",
        xaxis_title="Soil Clay (%)",
        yaxis_title="Water Balance (mm)",
    )
    return format_chart(fig, height=360)


def chart_q6(summary):
    """Q6: Geographic site coordinates, carbon content, and rainfall totals on dark carto tiles."""
    s = summary.copy()
    kwargs = dict(
        lat="lat",
        lon="lon",
        color="soc",
        size="rain_mm",
        hover_name="site",
        hover_data={"lat": ":.3f", "lon": ":.3f", "rain_mm": ":.0f", "soc": ":.1f", "clay": ":.1f"},
        color_continuous_scale="Tealgrn",
        title="Q6: Monitoring locations (Colour = SOC, Size = Rainfall)",
        labels={"soc": "SOC (g/kg)", "rain_mm": "Rain (mm)"},
    )
    try:
        fig = px.scatter_map(
            s,
            **kwargs,
            zoom=5.4,
            center={"lat": float(s["lat"].mean()), "lon": float(s["lon"].mean())},
            map_style="carto-darkmatter",
        )
    except Exception:
        fig = px.scatter_mapbox(
            s,
            **kwargs,
            zoom=5.4,
            center={"lat": float(s["lat"].mean()), "lon": float(s["lon"].mean())},
            mapbox_style="carto-darkmatter",
        )
    fig.update_layout(
        height=440,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12, color=COLORS["text"]),
    )
    return fig


# 7 Main Application Flow
def apply_page_style():
    """Apply high-contrast dark console layout styles."""
    st.set_page_config(
        page_title="ITD112 Laboratory Exercise 1: API Integration",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1200px;
            padding-top: 1.75rem;
            padding-bottom: 3.5rem;
        }
        /* Dark console metric cards */
        div[data-testid="stMetric"] {
            background-color: #151C22;
            border: 1px solid #27333F;
            padding: 0.8rem 1rem;
            border-radius: 6px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            color: #F1F5F9 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }
        /* Expander panel styling */
        div[data-testid="stExpander"] {
            background-color: #151C22;
            border: 1px solid #27333F;
            border-radius: 6px;
            margin-top: 0.5rem;
        }
        /* Primary button */
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #14B8A6;
            color: #0B0F12;
            font-weight: 700;
            border: none;
            height: 42px;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #2DD4BF;
            color: #0B0F12;
        }
        /* Subtle dividers */
        hr {
            border-color: #27333F;
            margin: 1.5rem 0;
        }
        /* Code text tint */
        code {
            color: #2DD4BF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_page_style()

    # Dark console page header
    st.title("Integrating Soil and Weather APIs")
    st.caption("ITD112: Laboratory Exercise 1 | Linking ISRIC SoilGrids 2.0 with Open-Meteo Historical Records")

    # Dark toolbar row
    c_sites, c_start, c_end, c_btn = st.columns([3, 1.5, 1.5, 1.5], vertical_alignment="bottom")
    all_sites = SITES["site"].tolist()
    chosen = c_sites.multiselect(
        "Observation Sites",
        all_sites,
        default=all_sites[:3],
        help="Select 5 sites to enable Q5 trend regression.",
    )
    start = c_start.date_input("Start Date", date(2025, 1, 1))
    end = c_end.date_input("End Date", date(2025, 12, 31))
    fetch = c_btn.button("Run Integration", type="primary", use_container_width=True)

    sites = SITES[SITES["site"].isin(chosen)].reset_index(drop=True)

    if fetch:
        if sites.empty:
            st.error("Select at least one site before running.")
            st.stop()
        if start > end:
            st.error("Start date must be on or before the end date.")
            st.stop()

        # Step 2 & 3: Requests & Telemetry
        weather_req = build_weather_request(sites, start, end)
        soil_reqs = [build_soil_request(r.site, r.lat, r.lon) for r in sites.itertuples()]

        status_rows = []
        try:
            t0 = time.perf_counter()
            wres = send(weather_req)
            elapsed = wres["elapsed_s"] if wres["elapsed_s"] is not None else round(time.perf_counter() - t0, 2)
            status_rows.append({
                "API Service": "Open-Meteo",
                "Target": "All selected sites",
                "HTTP": str(wres["status"]) if wres["status"] else "200 (Cache)",
                "From Cache": "Yes (Disk)" if wres["from_cache"] else "No (Network)",
                "Elapsed": f"{elapsed:.2f}s" if not wres["from_cache"] else "0.00s",
                "Bytes": f"{wres['bytes']:,}",
            })
        except requests.RequestException as exc:
            st.error(f"Open-Meteo request failed: {exc}")
            st.stop()

        soil_res = []
        for i, req in enumerate(soil_reqs):
            try:
                res = send(req)
            except requests.RequestException as exc:
                st.error(
                    f"SoilGrids failed for {req['label']}: {exc}. "
                    "The SoilGrids REST API is a public beta and is occasionally paused. "
                    "Please retry or use cached fixtures."
                )
                st.stop()
            soil_res.append(res)
            elapsed = res["elapsed_s"] if res["elapsed_s"] is not None else "0.00"
            status_rows.append({
                "API Service": "SoilGrids",
                "Target": req["label"],
                "HTTP": str(res["status"]) if res["status"] else "200 (Cache)",
                "From Cache": "Yes (Disk)" if res["from_cache"] else "No (Network)",
                "Elapsed": f"{elapsed}s" if not res["from_cache"] else "0.00s",
                "Bytes": f"{res['bytes']:,}",
            })
            if not res["from_cache"] and i < len(soil_reqs) - 1:
                time.sleep(SOILGRIDS_PAUSE_S)

        # Step 4: Schema Parsing
        weather = parse_weather(wres["data"], sites)
        soil_frames = [parse_soil(res["data"], req["label"]) for res, req in zip(soil_res, soil_reqs)]
        soil_long = pd.concat(soil_frames, ignore_index=True)
        soil = soil_table(soil_long, sites)

        # Step 5: Relational Integration
        daily, summary = integrate(weather, soil)
        assert len(daily) == len(weather), "Join changed the row count; verify site keys."

        missing = soil.loc[soil["clay"].isna(), "site"].tolist()

        # Save integrated dataset in session state so widget reruns preserve the view
        st.session_state["pipeline_results"] = {
            "weather_req": weather_req,
            "soil_reqs": soil_reqs,
            "status_rows": status_rows,
            "weather": weather,
            "soil_long": soil_long,
            "soil": soil,
            "daily": daily,
            "summary": summary,
            "sites": sites,
            "missing": missing,
        }

    if "pipeline_results" not in st.session_state:
        st.info("Select monitoring sites and observation dates above, then click 'Run Integration'.")
        st.stop()

    # Load active pipeline results
    pipeline = st.session_state["pipeline_results"]
    weather_req = pipeline["weather_req"]
    soil_reqs = pipeline["soil_reqs"]
    status_rows = pipeline["status_rows"]
    weather = pipeline["weather"]
    soil_long = pipeline["soil_long"]
    soil = pipeline["soil"]
    daily = pipeline["daily"]
    summary = pipeline["summary"]
    active_sites = pipeline["sites"]
    missing = pipeline["missing"]

    if missing:
        st.warning(
            f"SoilGrids returned null for: {', '.join(missing)}. "
            "The coordinate falls on water, built-up ground, or an unmapped grid. "
            "Try adjusting coordinates slightly inland."
        )

    # Section: Data Pipeline & Normalization
    st.markdown("---")
    st.subheader("Data Pipeline and Transmission")

    col_meta1, col_meta2 = st.columns([3, 2])
    with col_meta1:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
    with col_meta2:
        m1, m2 = st.columns(2)
        m1.metric("Daily Weather Rows", f"{len(weather):,}")
        m2.metric("Joined Total Rows", f"{len(daily):,}")
        m3, m4 = st.columns(2)
        m3.metric("Total Rainfall", f"{summary['rain_mm'].sum():,.0f} mm")
        surplus = int((summary["water_balance_mm"] >= 0).sum())
        m4.metric("Surplus Sites", f"{surplus} of {len(summary)}")

    with st.expander("Inspect Prepared Query URLs"):
        st.markdown("**Open-Meteo batched request URL:**")
        st.code(weather_req["full_url"], language="text")
        st.markdown("**SoilGrids per-site request URLs (repeated 'property' parameter):**")
        for req in soil_reqs:
            st.code(f"{req['label']}: {req['full_url']}", language="text")

    with st.expander("Inspect Normalized Data Tables"):
        tab_summary, tab_daily, tab_audit = st.tabs([
            "Site Summary Matrix",
            "Joined Daily Records",
            "Soil Scaling Audit (raw_mean / d_factor)",
        ])
        with tab_summary:
            st.dataframe(summary, use_container_width=True)
        with tab_daily:
            st.dataframe(daily.head(15), use_container_width=True)
        with tab_audit:
            st.dataframe(soil_long, use_container_width=True)

    # Section 6: Analytical Visualizations (Dark Dashboard Layout)
    st.markdown("---")
    st.subheader("Analytical Visualizations")

    # Q1: Full Width Time-Series
    st.plotly_chart(chart_q1(daily), use_container_width=True)

    # Row 1: Q2 & Q3 Paired
    col_q2, col_q3 = st.columns(2)
    with col_q2:
        site_options = active_sites["site"].tolist()
        if "q2_site_select" in st.session_state and st.session_state["q2_site_select"] not in site_options:
            st.session_state["q2_site_select"] = site_options[0]
        selected_q2 = st.selectbox(
            "Site for Q2 Demand Analysis",
            site_options,
            key="q2_site_select",
        )
        st.plotly_chart(chart_q2(daily, selected_q2), use_container_width=True)
    with col_q3:
        st.write("")  # Alignment spacer for selectbox
        st.write("")
        st.plotly_chart(chart_q3(daily), use_container_width=True)

    # Row 2: Q4 & Q5 Paired
    col_q4, col_q5 = st.columns(2)
    with col_q4:
        st.plotly_chart(chart_q4(summary), use_container_width=True)
    with col_q5:
        st.plotly_chart(chart_q5(summary), use_container_width=True)
        if len(summary) < 5:
            st.caption("Note: Trend line appears when 5 or more sites are selected.")

    # Row 3: Q6 Full Width Map (Dark Tiles)
    st.plotly_chart(chart_q6(summary), use_container_width=True)

    # Section 7: Export & Provenance
    st.markdown("---")
    st.subheader("Export and Provenance")
    b1, b2, _ = st.columns([1, 1, 2])
    b1.download_button(
        "Download Site Summary (CSV)",
        summary.to_csv(index=False),
        "site_summary.csv",
        "text/csv",
        use_container_width=True,
    )
    b2.download_button(
        "Download Daily Records (CSV)",
        daily.drop(columns="month").to_csv(index=False),
        "integrated_daily.csv",
        "text/csv",
        use_container_width=True,
    )

    st.caption(SOURCE_NOTE)


if __name__ == "__main__":
    main()
