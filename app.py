"""
Timber Forecasting Uncertainty Dashboard
=======================================

An interactive Streamlit demo dashboard that shows how measurement uncertainty
propagates through a deterministic timber-volume forecast via Monte Carlo
simulation.

Synthetic data only.

Run:
    streamlit run app.py

Dependencies:
    streamlit, pandas, numpy, plotly  (see requirements.txt)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Design tokens — forest / earthy palette
# =============================================================================
# Kept as a single source of truth so the Streamlit theme, custom CSS and
# Plotly figures all pull from the same values instead of drifting apart.

PRIMARY_DARK = "#1B4332"     # deep forest green — headings, total/median lines
MID_GREEN = "#2D6A4F"        # conifer green — primary accent, Norway spruce
SAGE = "#84A98C"             # muted sage — gridlines, secondary accents
WOOD_ACCENT = "#B08968"      # bark/wood tone — Oak, deterministic line
BG_PARCHMENT = "#F7F5EF"     # page background
BG_CARD = "#EFEBE0"          # card / secondary background
TEXT_CHARCOAL = "#283618"    # body text
GRID_COLOR = "#DDD6C4"       # chart gridlines, tuned to sit on parchment

SPECIES_COLORS = {"Norway spruce": MID_GREEN, "Oak": WOOD_ACCENT}
BAND_95_COLOR = "rgba(132, 169, 140, 0.20)"
BAND_80_COLOR = "rgba(45, 106, 79, 0.30)"

DISPLAY_FONT = "Fraunces"
BODY_FONT = "Inter"


# =============================================================================
# Page setup & styling
# =============================================================================

st.set_page_config(
    page_title="Timber Forecasting Uncertainty Dashboard",
    page_icon="🌲",
    layout="wide",
)


def tree_ring_svg(size: int = 60) -> str:
    """A small concentric-ring mark — doubles as a tree-ring / uncertainty-band motif."""
    c = size/2
    rings = [
        (c * 0.28, PRIMARY_DARK, 0.9),
        (c * 0.48, MID_GREEN, 0.8),
        (c * 0.68, SAGE, 0.6),
        (c * 0.88, WOOD_ACCENT, 0.45),
    ]
    circles = "".join(
        f'<circle cx="{c}" cy="{c}" r="{r:.1f}" fill="none" '
        f'stroke="{color}" stroke-width="1.6" opacity="{opacity}" />'
        for r, color, opacity in rings
    )
    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{circles}</svg>'


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

        .stApp {{
            background-color: {BG_PARCHMENT};
            font-family: '{BODY_FONT}', sans-serif;
            color: {TEXT_CHARCOAL};
        }}
        h1, h2, h3, h4 {{
            font-family: '{DISPLAY_FONT}', serif !important;
            color: {PRIMARY_DARK} !important;
            font-weight: 600 !important;
        }}

        /* Hero banner */
        .hero {{
            background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {MID_GREEN} 100%);
            border-radius: 14px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.75rem;
            display: flex;
            align-items: center;
            gap: 1.1rem;
        }}
        .hero-title {{
            font-family: '{DISPLAY_FONT}', serif;
            color: #FBF9F3;
            font-size: 1.9rem;
            font-weight: 600;
            margin: 0;
            line-height: 1.25;
        }}
        .hero-sub {{
            font-family: '{BODY_FONT}', sans-serif;
            color: #E4E9E1;
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }}
        .hero svg circle {{ stroke: #FBF9F3; }}

        /* Section divider with tree-ring mark */
        .ring-divider {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 1.75rem 0 1.1rem 0;
        }}
        .ring-divider .line {{
            flex: 1;
            height: 1px;
            background: {GRID_COLOR};
        }}
        .ring-divider .label {{
            font-family: '{DISPLAY_FONT}', serif;
            color: {PRIMARY_DARK};
            font-weight: 600;
            font-size: 1.15rem;
            white-space: nowrap;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background-color: {BG_CARD};
            border: 1px solid {GRID_COLOR};
            border-radius: 10px;
            padding: 0.9rem 1rem 0.7rem 1rem;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {PRIMARY_DARK};
            font-weight: 600;
        }}
        div[data-testid="stMetricValue"] {{
            color: {TEXT_CHARCOAL};
            font-family: '{DISPLAY_FONT}', serif;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {MID_GREEN};
            color: #FBF9F3;
            border: none;
            border-radius: 8px;
            font-weight: 500;
        }}
        .stButton > button:hover {{
            background-color: {PRIMARY_DARK};
            color: #FBF9F3;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {BG_CARD};
        }}

        /* Footer caption */
        .footnote {{
            color: #6B6558;
            font-size: 0.82rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            {tree_ring_svg(60)}
            <div>
                <p class="hero-title">{title}</p>
                <p class="hero-sub">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_divider(label: str) -> None:
    st.markdown(
        f"""
        <div class="ring-divider">
            <div class="line"></div>
            {tree_ring_svg(28)}
            <div class="label">{label}</div>
            <div class="line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 440, title: str | None = None) -> go.Figure:
    """Apply the shared Plotly theme so every chart in the app reads as one family."""
    fig.update_layout(
        height=height,
        font=dict(family=f"{BODY_FONT}, sans-serif", color=TEXT_CHARCOAL, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Larger top margin keeps horizontal legends from colliding with titles.
        margin=dict(l=10, r=10, t=96 if title else 28, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", font_family=f"{BODY_FONT}, sans-serif", font_color=TEXT_CHARCOAL),
        title=(
            dict(
                text=title,
                y=0.98,
                yanchor="top",
                font=dict(family=f"{DISPLAY_FONT}, serif", size=17, color=PRIMARY_DARK),
            )
            if title else None
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    return fig

# =============================================================================
# Demo region map (geometry helpers)
# =============================================================================

APP_DIR = Path(__file__).resolve().parent
GEOJSON_PATH = APP_DIR / "data" / "region-grand-est.geojson"


def _geometry_to_rings(geometry: dict) -> list[np.ndarray]:
    """Return exterior rings as arrays of [lon, lat] from a GeoJSON geometry."""
    if not geometry:
        return []

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if geom_type == "Polygon":
        if coords:
            return [np.asarray(coords[0], dtype=float)]
        return []

    if geom_type == "MultiPolygon":
        rings = []
        for polygon in coords or []:
            if polygon:
                rings.append(np.asarray(polygon[0], dtype=float))
        return rings

    if geom_type == "GeometryCollection":
        rings = []
        for subgeom in geometry.get("geometries", []):
            rings.extend(_geometry_to_rings(subgeom))
        return rings

    return []


def load_geojson_rings(path: Path) -> list[np.ndarray]:
    """Load exterior polygon rings from a GeoJSON FeatureCollection/Feature/Geometry."""
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    obj_type = obj.get("type")

    if obj_type == "FeatureCollection":
        rings = []
        for feature in obj.get("features", []):
            rings.extend(_geometry_to_rings(feature.get("geometry", {})))
        return rings

    if obj_type == "Feature":
        return _geometry_to_rings(obj.get("geometry", {}))

    return _geometry_to_rings(obj)


# Fallback approximate, simplified polygon around a Grand Est-style demo region.
GRAND_EST_DEMO_POLYGON = np.asarray(
    [
        [3.40, 48.10], [3.70, 49.45], [4.80, 50.05], [6.20, 49.70],
        [7.65, 49.20], [7.70, 48.05], [7.10, 47.55], [5.75, 47.55],
        [4.25, 47.85], [3.40, 48.10],
    ],
    dtype=float,
)

BOX_COORDS = {
    "Box 1": {"lat": 48.58, "lon": 7.75},
    "Box 2": {"lat": 48.29, "lon": 7.42},
    "Box 3": {"lat": 48.08, "lon": 7.36},
    "Box 4": {"lat": 48.69, "lon": 6.18},
    "Box 5": {"lat": 49.12, "lon": 6.17},
    "Box 6": {"lat": 48.45, "lon": 5.95},
    "Box 7": {"lat": 48.30, "lon": 4.10},
    "Box 8": {"lat": 49.26, "lon": 4.03},
    "Box 9": {"lat": 48.95, "lon": 5.15},
    "Box 10": {"lat": 47.93, "lon": 6.85},
}

MAP_BOXES = pd.DataFrame([{"forest_box": box, **coords} for box, coords in BOX_COORDS.items()])


def load_region_rings() -> tuple[list[np.ndarray], str]:
    try:
        rings = load_geojson_rings(GEOJSON_PATH)
        label = f"Loaded real GeoJSON boundary: `{GEOJSON_PATH.name}`"
        return rings, label
    except Exception as exc:
        label = (
            "Using fallback synthetic boundary because `data/region-grand-est.geojson` "
            f"could not be loaded. Details: {exc}"
        )
        return [GRAND_EST_DEMO_POLYGON], label


def _synthetic_patch_ring(lon: float, lat: float, area_ha: float, index: int) -> np.ndarray:
    """Create a small irregular forest patch around a map point for demo visualisation."""
    angles = np.linspace(0, 2 * math.pi, 18, endpoint=False)
    # Size is deliberately visual rather than geographic: larger input area means a larger patch.
    base = 0.055 + 0.012 * math.sqrt(max(float(area_ha), 1.0))
    wobble = 1.0 + 0.12 * np.sin(3 * angles + index) + 0.06 * np.cos(5 * angles + 0.7 * index)
    lon_radius = base * wobble
    lat_radius = base * 0.62 * wobble
    ring = np.column_stack([lon + lon_radius * np.cos(angles), lat + lat_radius * np.sin(angles)])
    return np.vstack([ring, ring[0]])


def build_region_map(
    region_rings: list[np.ndarray], box_df: pd.DataFrame, forest_df: pd.DataFrame | None = None
) -> go.Figure:
    """Interactive, pannable/zoomable region map (no Mapbox token required)."""
    fig = go.Figure()

    for ring in region_rings:
        if ring.size == 0 or ring.shape[1] < 2:
            continue
        fig.add_trace(
            go.Scattermapbox(
                lon=ring[:, 0], lat=ring[:, 1],
                mode="lines", fill="toself",
                fillcolor="rgba(45, 106, 79, 0.12)",
                line=dict(width=2, color=MID_GREEN),
                hoverinfo="skip", showlegend=False,
            )
        )

    if forest_df is not None and not forest_df.empty:
        box_summary = (
            forest_df.groupby("forest_box")
            .agg(
                area_ha=("area_ha", "sum"),
                tree_groups=("tree_group_id", "nunique"),
                species=("species", lambda x: ", ".join(sorted(set(map(str, x))))),
            )
            .reset_index()
        )
        map_df = box_df.merge(box_summary, on="forest_box", how="left")
    else:
        map_df = box_df.copy()
        map_df["area_ha"] = 8.0
        map_df["tree_groups"] = 1
        map_df["species"] = "Synthetic"

    for idx, row in map_df.reset_index(drop=True).iterrows():
        ring = _synthetic_patch_ring(row["lon"], row["lat"], row["area_ha"], idx)
        hover = (
            f"{row['forest_box']}<br>"
            f"Area: {row['area_ha']:.1f} ha<br>"
            f"Tree groups: {int(row['tree_groups'])}<br>"
            f"Species: {row['species']}"
        )
        fig.add_trace(
            go.Scattermapbox(
                lon=ring[:, 0],
                lat=ring[:, 1],
                mode="lines",
                fill="toself",
                fillcolor="rgba(45, 106, 79, 0.24)",
                line=dict(width=1.5, color=PRIMARY_DARK),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lon=map_df["lon"], lat=map_df["lat"],
            mode="markers+text",
            marker=dict(size=18, color=WOOD_ACCENT, opacity=0.90),
            text=map_df["forest_box"],
            textposition="top right",
            textfont=dict(size=11, color=PRIMARY_DARK, family=f"{BODY_FONT}, sans-serif"),
            hovertext=map_df["forest_box"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=float(box_df["lat"].mean()), lon=float(box_df["lon"].mean())),
            zoom=6.0,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# =============================================================================
# Growth / volume model
# =============================================================================

@dataclass(frozen=True)
class SpeciesParams:
    display_name: str
    group: str
    h_max: float
    k_base: float
    shape_m: float
    form_factor: float
    dbh_height_elasticity: float


SPECIES_PARAMS: dict[str, SpeciesParams] = {
    "Norway spruce": SpeciesParams(
        display_name="Norway spruce", group="Conifer",
        h_max=39.0, k_base=0.030, shape_m=1.55,
        form_factor=0.45, dbh_height_elasticity=0.72,
    ),
    "Oak": SpeciesParams(
        display_name="Oak", group="Broadleaf",
        h_max=32.0, k_base=0.022, shape_m=1.35,
        form_factor=0.48, dbh_height_elasticity=0.58,
    ),
}


@st.cache_data
def make_synthetic_forest_boxes(start_year: int) -> pd.DataFrame:
    """Create a small synthetic forest-region input table.

    `start_year` is a real cache key here (not a closed-over global), so the
    cache correctly invalidates whenever the sidebar's start year changes.
    """
    data = [
        # box, tree group, species, area, planting, yield class, top height, top height sd, DBH, stems
        ("Box 1", "B1-A", "Norway spruce", 8.5, 1982, 9, 24.3, 1.2, 31.0, 360),
        ("Box 1", "B1-B", "Oak",           3.0, 1971, 6, 21.6, 0.9, 35.0, 190),
        ("Box 2", "B2-A", "Norway spruce", 10.2, 1995, 8, 19.7, 1.0, 26.0, 410),
        ("Box 3", "B3-A", "Oak",           6.8, 1965, 7, 24.1, 1.1, 39.0, 165),
        ("Box 3", "B3-B", "Norway spruce", 4.2, 2005, 7, 14.9, 1.3, 21.0, 500),
        ("Box 4", "B4-A", "Norway spruce", 12.5, 2012, 6, 10.8, 0.8, 16.5, 620),
        ("Box 5", "B5-A", "Oak",           7.7, 1988, 5, 18.9, 0.7, 29.5, 210),
        ("Box 6", "B6-A", "Norway spruce", 9.4, 2001, 8, 17.5, 1.2, 24.0, 460),
        ("Box 6", "B6-B", "Oak",           2.8, 1978, 6, 22.3, 0.9, 33.5, 175),
        ("Box 7", "B7-A", "Norway spruce", 11.0, 1990, 10, 22.5, 1.4, 29.0, 385),
        ("Box 8", "B8-A", "Oak",           5.5, 1958, 8, 25.7, 1.0, 42.0, 145),
        ("Box 9", "B9-A", "Norway spruce", 13.2, 2018, 5, 7.9, 0.6, 12.5, 780),
        ("Box 10", "B10-A", "Norway spruce", 7.5, 2008, 7, 13.2, 0.9, 19.0, 540),
        ("Box 10", "B10-B", "Oak",          2.5, 1998, 5, 17.1, 0.8, 27.0, 220),
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "forest_box", "tree_group_id", "species", "area_ha", "planting_year",
            "yield_class", "top_height_m", "top_height_sd_m", "dbh_cm", "stems_per_ha",
        ],
    )
    df["region"] = "Grand Est synthetic demo region"
    df["tree_group"] = df["forest_box"] + " / " + df["tree_group_id"]
    df["latitude"] = df["forest_box"].map(lambda b: BOX_COORDS[b]["lat"])
    df["longitude"] = df["forest_box"].map(lambda b: BOX_COORDS[b]["lon"])
    df["age_start"] = start_year - df["planting_year"]
    df["starting_volume_m3_ha"] = standing_volume_per_ha(
        df["species"], df["top_height_m"], df["dbh_cm"], df["stems_per_ha"]
    )
    df["starting_volume_m3"] = df["starting_volume_m3_ha"] * df["area_ha"]
    return df


def richards_top_height(age: np.ndarray | float, species: str, yield_class: int) -> np.ndarray | float:
    """
    Chapman-Richards-style top-height curve: H(age) = Hmax * (1 - exp(-k * age))^m

    A toy growth curve. Yield class changes k, so higher productivity reaches
    height faster.
    """
    p = SPECIES_PARAMS[species]
    age_arr = np.maximum(np.asarray(age, dtype=float), 0.0)
    yc_factor = 0.80 + 0.055 * float(yield_class)
    k = p.k_base * yc_factor
    return p.h_max * np.power(1.0 - np.exp(-k * age_arr), p.shape_m)


def future_top_height_from_current(
    current_height: float, current_age: float, future_age: float,
    species: str, yield_class: int,
) -> float:
    """Use Richards curve increment from current age to future age, anchored to observed current height."""
    curve_now = float(richards_top_height(current_age, species, yield_class))
    curve_future = float(richards_top_height(future_age, species, yield_class))
    increment = max(curve_future - curve_now, 0.0)
    return max(current_height + increment, 0.1)


def future_dbh_from_height(
    current_dbh_cm: float, current_height_m: float, future_height_m: float, species: str,
) -> float:
    """Simple DBH response to top-height growth."""
    p = SPECIES_PARAMS[species]
    ratio = max(future_height_m, 0.1) / max(current_height_m, 0.1)
    return max(current_dbh_cm * ratio ** p.dbh_height_elasticity, 0.1)


def standing_volume_per_ha(
    species: pd.Series | np.ndarray | list[str],
    top_height_m: pd.Series | np.ndarray,
    dbh_cm: pd.Series | np.ndarray,
    stems_per_ha: pd.Series | np.ndarray,
) -> np.ndarray:
    """Approximate stand volume using tree form factor × basal area × height × stems."""
    species_arr = np.asarray(species)
    h = np.maximum(np.asarray(top_height_m, dtype=float), 0.1)
    dbh_m = np.maximum(np.asarray(dbh_cm, dtype=float), 0.1) / 100.0
    stems = np.maximum(np.asarray(stems_per_ha, dtype=float), 0.0)
    form_factors = np.array([SPECIES_PARAMS[str(s)].form_factor for s in species_arr], dtype=float)
    basal_area_tree = math.pi * (dbh_m / 2.0) ** 2
    tree_volume = form_factors * basal_area_tree * h
    return tree_volume * stems


def forecast_tree_groups(input_df: pd.DataFrame, years: np.ndarray, start_year: int) -> pd.DataFrame:
    """Run deterministic forecast for every synthetic tree group."""
    rows = []
    for _, row in input_df.iterrows():
        species = row["species"]
        yield_class = int(row["yield_class"])
        area_ha = float(row["area_ha"])
        planting_year = int(row["planting_year"])
        current_age = max(start_year - planting_year, 1)
        current_height = float(row["top_height_m"])
        current_dbh = float(row["dbh_cm"])
        stems = float(row["stems_per_ha"])

        for year in years:
            age = max(year - planting_year, 1)
            height = future_top_height_from_current(current_height, current_age, age, species, yield_class)
            dbh = future_dbh_from_height(current_dbh, current_height, height, species)
            volume_m3_ha = standing_volume_per_ha([species], [height], [dbh], [stems])[0]
            rows.append(
                {
                    "year": int(year),
                    "region": row["region"],
                    "forest_box": row["forest_box"],
                    "tree_group_id": row["tree_group_id"],
                    "species": species,
                    "species_group": SPECIES_PARAMS[species].group,
                    "area_ha": area_ha,
                    "yield_class": yield_class,
                    "top_height_m": height,
                    "dbh_cm": dbh,
                    "volume_m3_ha": volume_m3_ha,
                    "volume_m3": volume_m3_ha * area_ha,
                }
            )
    return pd.DataFrame(rows)


def aggregate_forecast(forecast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_species = forecast_df.groupby(["year", "species"], as_index=False)["volume_m3"].sum()
    total = forecast_df.groupby("year", as_index=False)["volume_m3"].sum()
    return by_species, total


def run_monte_carlo(
    input_df: pd.DataFrame, years: np.ndarray, n: int, seed_value: int,
    start_year: int, top_height_sigma_m: float, dbh_sigma_cm: float,
) -> pd.DataFrame:
    """Sample top height and DBH uncertainty, run forecast, and aggregate by species and total."""
    rng = np.random.default_rng(seed_value)
    all_summaries = []

    th_sigma = np.maximum(input_df["top_height_sd_m"].to_numpy(dtype=float), 0.0)
    # Scale component-specific uncertainty by sidebar factor relative to the synthetic default.
    th_sigma = th_sigma * (top_height_sigma_m / 1.0)

    for sim_id in range(n):
        sample = input_df.copy()
        sample["top_height_m"] = rng.normal(
            loc=input_df["top_height_m"].to_numpy(dtype=float), scale=th_sigma
        ).clip(0.1, None)
        sample["dbh_cm"] = rng.normal(
            loc=input_df["dbh_cm"].to_numpy(dtype=float), scale=dbh_sigma_cm
        ).clip(1.0, None)

        fc = forecast_tree_groups(sample, years, start_year)

        by_species = fc.groupby(["year", "species"], as_index=False)["volume_m3"].sum()
        by_species = by_species.rename(columns={"species": "series"})

        total = fc.groupby("year", as_index=False)["volume_m3"].sum()
        total["series"] = "Total"

        sim_summary = pd.concat(
            [
                by_species[["year", "series", "volume_m3"]],
                total[["year", "series", "volume_m3"]],
            ],
            ignore_index=True,
        )
        sim_summary["simulation"] = sim_id
        all_summaries.append(sim_summary)

    return pd.concat(all_summaries, ignore_index=True)


def deterministic_series_table(by_species: pd.DataFrame, total_forecast: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic forecast in the same long format as the Monte Carlo output."""
    species_series = by_species.rename(columns={"species": "series"})[["year", "series", "volume_m3"]]
    total_series = total_forecast.copy()
    total_series["series"] = "Total"
    total_series = total_series[["year", "series", "volume_m3"]]
    return pd.concat([species_series, total_series], ignore_index=True)


def summarize_simulations(sims: pd.DataFrame) -> pd.DataFrame:
    """Summarise Monte Carlo forecasts for each plotted series."""
    return (
        sims.groupby(["series", "year"])["volume_m3"]
        .agg(
            median="median",
            p10=lambda x: np.percentile(x, 10),
            p90=lambda x: np.percentile(x, 90),
            p2_5=lambda x: np.percentile(x, 2.5),
            p97_5=lambda x: np.percentile(x, 97.5),
        )
        .reset_index()
    )

# =============================================================================
# Chart builders
# =============================================================================

def build_species_area_chart(species_area: pd.DataFrame) -> go.Figure:
    colors = [SPECIES_COLORS.get(s, MID_GREEN) for s in species_area["species"]]
    max_area = float(species_area["area_ha"].max()) if not species_area.empty else 1.0

    fig = go.Figure(
        go.Bar(
            x=species_area["species"], y=species_area["area_ha"],
            marker_color=colors, text=species_area["area_ha"].round(1),
            texttemplate="%{text:.1f} ha", textposition="outside",
            textfont=dict(size=13, color=TEXT_CHARCOAL),
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.1f} ha<extra></extra>",
        )
    )
    fig.update_layout(yaxis_title="Area (ha)")
    fig = style_figure(fig, height=360, title="Synthetic species area")
    fig.update_yaxes(range=[0, max_area * 1.22])
    fig.update_layout(margin=dict(l=10, r=10, t=75, b=10))
    return fig


def build_deterministic_forecast_chart(by_species: pd.DataFrame, total_forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for species, grp in by_species.groupby("species"):
        fig.add_trace(
            go.Scatter(
                x=grp["year"], y=grp["volume_m3"], mode="lines", name=species,
                line=dict(width=2.5, color=SPECIES_COLORS.get(species, MID_GREEN)),
                hovertemplate="%{fullData.name}<br>%{x}: %{y:,.0f} m³<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=total_forecast["year"], y=total_forecast["volume_m3"], mode="lines", name="Total",
            line=dict(width=3, color=PRIMARY_DARK, dash="dash"),
            hovertemplate="Total<br>%{x}: %{y:,.0f} m³<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Standing volume (m³)", hovermode="x unified")
    return style_figure(fig, height=460, title="Deterministic regional timber-volume forecast")


def build_uncertainty_chart(
    deterministic_series: pd.DataFrame, sim_summary: pd.DataFrame, series_name: str
) -> go.Figure:
    """Plot uncertainty for one series, so deterministic and simulated quantities match."""
    det = deterministic_series[deterministic_series["series"] == series_name]
    summary = sim_summary[sim_summary["series"] == series_name]

    fig = go.Figure()

    # 95% band
    fig.add_trace(go.Scatter(x=summary["year"], y=summary["p97_5"], mode="lines",
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=summary["year"], y=summary["p2_5"], mode="lines",
                              line=dict(width=0), fill="tonexty", fillcolor=BAND_95_COLOR,
                              name="95% simulation band", hoverinfo="skip"))
    # 80% band
    fig.add_trace(go.Scatter(x=summary["year"], y=summary["p90"], mode="lines",
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=summary["year"], y=summary["p10"], mode="lines",
                              line=dict(width=0), fill="tonexty", fillcolor=BAND_80_COLOR,
                              name="80% simulation band", hoverinfo="skip"))

    line_color = PRIMARY_DARK if series_name == "Total" else SPECIES_COLORS.get(series_name, WOOD_ACCENT)

    fig.add_trace(go.Scatter(x=det["year"], y=det["volume_m3"], mode="lines",
                              name=f"Deterministic {series_name}",
                              line=dict(width=2.7, color=line_color),
                              hovertemplate=f"Deterministic {series_name}<br>%{{x}}: %{{y:,.0f}} m³<extra></extra>"))
    fig.add_trace(go.Scatter(x=summary["year"], y=summary["median"], mode="lines",
                              name=f"Median simulated {series_name}",
                              line=dict(width=2.7, color=WOOD_ACCENT, dash="dash"),
                              hovertemplate=f"Median {series_name}<br>%{{x}}: %{{y:,.0f}} m³<extra></extra>"))

    fig.update_layout(xaxis_title="Year", yaxis_title="Standing volume (m³)", hovermode="x unified")
    return style_figure(
        fig,
        height=500,
        title=f"Timber-volume forecast with uncertainty · {series_name}",
    )

# =============================================================================
# App
# =============================================================================

def render_sidebar() -> dict:
    st.sidebar.header("Dashboard controls")

    start_year = int(st.sidebar.number_input("Forecast start year", min_value=2020, max_value=2040, value=2026, step=1))
    end_year = int(st.sidebar.number_input("Forecast end year", min_value=start_year + 5, max_value=2100, value=2055, step=1))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Measurement uncertainty")
    top_height_sigma_m = float(
        st.sidebar.slider(
            "Top-height uncertainty (m)", min_value=0.0, max_value=3.0, value=1.0, step=0.1,
            help="Synthetic measurement/input uncertainty applied to each tree-group top height.",
        )
    )
    dbh_sigma_cm = float(
        st.sidebar.slider(
            "DBH uncertainty (cm)", min_value=0.0, max_value=4.0, value=1.0, step=0.1,
            help="Synthetic measurement/input uncertainty applied to mean diameter at breast height.",
        )
    )

    n_sim = int(st.sidebar.slider("Monte Carlo simulations", min_value=100, max_value=3000, value=800, step=100))
    seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1))

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "The equations are deliberately simple and synthetic. They are not an official "
        "inventory or production forecast model."
    )

    return dict(
        start_year=start_year, end_year=end_year,
        top_height_sigma_m=top_height_sigma_m, dbh_sigma_cm=dbh_sigma_cm,
        n_sim=n_sim, seed=seed,
    )


def main() -> None:
    inject_css()
    hero(
        "Timber forecasting",
        "Turning a deterministic forest timber forecast into an uncertainty band",
    )

    params = render_sidebar()
    start_year, end_year = params["start_year"], params["end_year"]
    years = np.arange(start_year, end_year + 1)

    # ---- Data prep -----------------------------------------------------------
    forest = make_synthetic_forest_boxes(start_year)
    forecast = forecast_tree_groups(forest, years, start_year)
    by_species, total_forecast = aggregate_forecast(forecast)
    deterministic_series = deterministic_series_table(by_species, total_forecast)

    # ---- Intro + map -------------------------------------------------------
    intro_col, map_col = st.columns([1.35, 1.0], gap="large")

    with intro_col:
        st.subheader("Dashboard overview")
        st.markdown(
            """
This is a **demo dashboard** for a deterministic timber forecast. The starting point is
one synthetic forest input table, which produces one future timber-volume curve. The
dashboard then adds uncertainty to the measurements that feed the forecast and runs
many simulations. The result is an uncertainty band rather than a single forecast line.

The example uses a synthetic **Grand Est forest region in France**. The region is split
into simplified **forest units**: small forested patches that behave like compartment-
or stand-level modelling units. In a real inventory workflow, these units may be
separated by management boundaries, roads, species changes, age structure or other
mapping rules. Each unit can contain one or more tree groups, such as Norway spruce
or oak, with different ages, top heights, DBH values and starting volumes.

**Forestry terms used below:** DBH, standing volume, stem density and yield class are
standard forestry terms. Useful references are the
[Forest Research technical glossary](https://cdn.forestresearch.gov.uk/2022/04/PF2011_Technical_Glossary_GLm8Sgl.pdf)
and the [Forest Research Forest Yield guide](https://www.forestresearch.gov.uk/tools-and-resources/fthr/forest-yield/).

### Workflow
1. Create a small synthetic forest input table.
2. Run a deterministic timber forecast.
3. Add uncertainty to top height and DBH.
4. Run Monte Carlo simulations.
5. Compare deterministic and uncertain forecasts for Norway spruce, oak and the total.
            """
        )

    with map_col:
        st.subheader("Grand Est demo forest units")
        region_rings, map_source_label = load_region_rings()

        if GEOJSON_PATH.exists():
            st.caption("Grand Est boundary loaded from your local GeoJSON file. Forest units and tree groups remain synthetic.")
        else:
            st.caption("Fallback synthetic boundary for orientation only. Add `data/region-grand-est.geojson` to use a real region boundary.")

        st.plotly_chart(
            build_region_map(region_rings, MAP_BOXES, forest),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        with st.expander("Map data note", expanded=False):
            st.write(map_source_label)
            st.write(
                "The green patches are synthetic forest units drawn for visual explanation. "
                "They are not official inventory boundaries. The measurements and forecast outputs are fully synthetic."
            )

    # ---- Section 1: input table ---------------------------------------------
    section_divider("1 · Synthetic forest-region input")
    st.markdown(
        "The region is divided into **forest units**. A unit is a simplified forested area used "
        "for the forecast, similar to a compartment or stand-level modelling unit. Each unit can "
        "contain one or more tree groups with a species, area, planting year, productivity class, "
        "top height, DBH, stem density and starting standing volume."
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Region", "Grand Est")
    metric_cols[1].metric("Forest units", forest["forest_box"].nunique())
    metric_cols[2].metric("Tree groups", len(forest))
    metric_cols[3].metric("Total area", f"{forest['area_ha'].sum():.1f} ha")
    metric_cols[4].metric("Starting volume", f"{forest['starting_volume_m3'].sum():,.0f} m³")

    summary_left, summary_right = st.columns([1.25, 1])

    with summary_left:
        display_cols = [
            "forest_box", "tree_group_id", "species", "area_ha", "planting_year", "age_start",
            "yield_class", "top_height_m", "top_height_sd_m", "dbh_cm", "stems_per_ha",
            "starting_volume_m3_ha", "starting_volume_m3",
        ]
        display_forest = forest[display_cols].rename(
            columns={
                "forest_box": "forest_unit",
                "tree_group_id": "tree_group",
            }
        )
        st.dataframe(
            display_forest.style.format(
                {
                    "area_ha": "{:.1f}",
                    "top_height_m": "{:.1f}",
                    "top_height_sd_m": "{:.1f}",
                    "dbh_cm": "{:.1f}",
                    "starting_volume_m3_ha": "{:.1f}",
                    "starting_volume_m3": "{:.0f}",
                }
            ),
            width="stretch", hide_index=True,
        )

    with summary_right:
        species_area = forest.groupby("species", as_index=False)["area_ha"].sum()
        st.plotly_chart(build_species_area_chart(species_area), use_container_width=True, config={"displayModeBar": False})

        species_start = forest.groupby("species", as_index=False)["starting_volume_m3"].sum()
        for _, row in species_start.iterrows():
            st.write(f"**{row['species']} starting volume:** {row['starting_volume_m3']:,.0f} m³")

    with st.expander("Model note: what is being forecast?", expanded=False):
        st.markdown(
            """
This demo uses a deliberately simple timber-volume model:

1. A Chapman–Richards-style curve grows top height through time.
2. DBH responds to top-height growth.
3. Standing volume is approximated from form factor × basal area × height × stem density.
4. Volumes are multiplied by area and summed across tree groups to give regional timber volume.

This is a teaching model only. It is not an official timber forecast model.
            """
        )

    # ---- Section 2: deterministic forecast -----------------------------------
    section_divider("2 · Deterministic forecast")
    st.markdown(
        "First we run the forecast once using the input table exactly as given. This is the "
        "deterministic forecast: one input table gives one future timber-volume curve for each "
        "species and for the regional total."
    )

    plot_col, end_col = st.columns([1.55, 1])

    with plot_col:
        st.plotly_chart(build_deterministic_forecast_chart(by_species, total_forecast), use_container_width=True, config={"displayModeBar": False})

    with end_col:
        end_values = by_species[by_species["year"] == end_year].copy()
        total_end = float(total_forecast.loc[total_forecast["year"] == end_year, "volume_m3"].iloc[0])
        st.subheader(f"Forecast volume in {end_year}")
        for _, row in end_values.iterrows():
            st.metric(row["species"], f"{row['volume_m3']:,.0f} m³")
        st.metric("Total", f"{total_end:,.0f} m³")

    # ---- Section 3: uncertainty propagation -----------------------------------
    section_divider("3 · Add measurement uncertainty and propagate it")
    st.markdown(
        "Now we add uncertainty to the measurements that influence the forecast, especially "
        "**top height** and **DBH**. For each simulation, the dashboard draws a possible top "
        "height and DBH for every tree group, runs the same forecast, and aggregates the result. "
        "The uncertainty charts below compare like with like: Norway spruce against Norway spruce, "
        "oak against oak, and total against total."
    )

    run_unc = st.button("▶  Run uncertainty model", type="primary")

    if run_unc:
        with st.spinner("Running Monte Carlo simulations"):
            sims = run_monte_carlo(
                forest, years, params["n_sim"], params["seed"],
                start_year, params["top_height_sigma_m"], params["dbh_sigma_cm"],
            )
            sim_summary = summarize_simulations(sims)

        series_order = ["Total"] + sorted(s for s in sim_summary["series"].unique() if s != "Total")
        tabs = st.tabs(series_order)

        for tab, series_name in zip(tabs, series_order):
            with tab:
                st.plotly_chart(
                    build_uncertainty_chart(deterministic_series, sim_summary, series_name),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                end_unc = sim_summary.loc[
                    (sim_summary["series"] == series_name) & (sim_summary["year"] == end_year)
                ].iloc[0]

                st.subheader(f"Uncertainty summary for {series_name} in {end_year}")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Median simulated volume", f"{end_unc['median']:,.0f} m³")
                col_b.metric("80% range", f"{end_unc['p10']:,.0f} – {end_unc['p90']:,.0f} m³")
                col_c.metric("95% range", f"{end_unc['p2_5']:,.0f} – {end_unc['p97_5']:,.0f} m³")

        with st.expander("Show simulated forecast summary table", expanded=False):
            st.dataframe(
                sim_summary.sort_values(["series", "year"]),
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("Click **Run uncertainty model** to generate the Monte Carlo forecast bands.")

    # ---- Footer ---------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<p class="footnote">Synthetic demo. All values are invented for explanation. The example is designed to '
        "explain uncertainty propagation, not to reproduce any operational forest inventory or "
        "official forecast system.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
