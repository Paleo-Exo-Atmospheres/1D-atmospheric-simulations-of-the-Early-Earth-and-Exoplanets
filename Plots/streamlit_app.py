"""Early Earth ozone explorer.

Interactive companion to "Simulations of the Evolving Ozone Layer: Implications
for the Early Earth, Exoplanets, and the Search For Life".

Run from this folder:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import early_earth_data as data

st.set_page_config(
    page_title="Evolving ozone layer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOURS = {
    "WACCM6": "#111111",
    "Atmos": "#ea580c",
    "Photochem": "#1d4ed8",
    "VULCAN": "#6b21a8",
    "VULCAN NCHO": "#6b21a8",
    "VULCAN SNCHOAr": "magenta",
    "Kasting": "#0f766e",
}

# Transmission spectra match the paper figures: solid lines, matplotlib colours.
# A single VULCAN curve is magenta. When both networks are drawn, the network
# without sulfur is dark purple and the sulfur network stays magenta.
SPECTRUM_COLOURS = {
    "WACCM6": "black",
    "Atmos": "darkorange",
    "Photochem": "blue",
    "VULCAN": "magenta",
    "VULCAN SNCHOAr": "magenta",
    "Kasting": "teal",
}
SPECTRUM_NO_S = "#6b21a8"

PAL_COLOURS = {
    "150": "#7f1d1d",
    "100": "#111111",
    "50": "#1d4ed8",
    "10": "#0f766e",
    "5": "#65a30d",
    "1": "#c2410c",
    "0.5": "#7c3aed",
    "0.1": "#db2777",
}

STYLE_COMPARE = "Compare models"
STYLE_OZONE = "Ozone column vs oxygen"
STYLE_MAP = "WACCM latitude–pressure"
STYLE_SPECTRUM = "Transmission spectra"
EARTH_WAVE_MIN = 0.2
EARTH_WAVE_MAX = 20.0

O3_BAND = "rgba(22, 101, 52, 0.18)"
O2_BAND = "rgba(29, 78, 216, 0.16)"
# x0, x1, label, fill, colour, label x (µm), label height as a fraction of the panel.
# The two visible O₂ bands share one label. O₂–X sits at 6.7 µm, below the O₃ label at 5.8 µm.
BANDS = (
    (0.20, 0.33, "O₃", O3_BAND, "#166534", None, 0.96),
    (0.48, 1.20, "O₃", O3_BAND, "#166534", None, 0.96),
    (4.65, 5.00, "O₃", O3_BAND, "#166534", None, 0.96),
    (5.70, 5.90, "O₃", O3_BAND, "#166534", 5.80, 0.96),
    (8.20, 10.20, "O₃", O3_BAND, "#166534", None, 0.96),
    (0.67, 0.70, "O₂", O2_BAND, "#1d4ed8", 0.73, 0.78),
    (0.75, 0.78, None, O2_BAND, "#1d4ed8", None, 0.78),
    (1.25, 1.30, "O₂", O2_BAND, "#1d4ed8", None, 0.78),
    (5.30, 7.00, "O₂–X", O2_BAND, "#1d4ed8", 6.70, 0.78),
)

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"], [data-testid="stBottom"] {
        background-color: #ffffff;
        color: #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #9CD6CB;
    }
    h1, h2, h3, p, label, span,
    [data-testid="stHeading"] h1,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #000000;
    }
    [data-testid="stPlotlyChart"] {
        background-color: #ffffff;
    }
    [data-testid="stSelectbox"] div:has(input),
    [data-testid="stMultiSelect"] div:has(input) {
        background-color: #f2f2f2 !important;
        color: #000000 !important;
        border-color: #d9d9d9 !important;
    }
    [data-testid="stSelectbox"] div:has(input) span,
    [data-testid="stSelectbox"] div:has(input) input,
    [data-testid="stMultiSelect"] div:has(input) input,
    [data-testid="stSelectbox"] svg,
    [data-testid="stMultiSelect"] svg {
        color: #000000 !important;
        fill: #000000 !important;
    }
    [role="listbox"],
    [role="option"] {
        background-color: #f3f3f3 !important;
        color: #000000 !important;
    }
    [role="option"][aria-selected="true"],
    [role="option"][data-hovered],
    [role="option"][data-focused],
    [role="option"][aria-selected="true"] [data-item-hl],
    [role="option"][data-hovered] [data-item-hl],
    [role="option"][data-focused] [data-item-hl] {
        background-color: #d6ebf5 !important;
        color: #000000 !important;
    }
    [data-testid="stMultiSelect"] span:has(button),
    [data-baseweb="tag"] {
        background-color: #d6ebf5 !important;
        color: #000000 !important;
        border-color: #b7d4e8 !important;
    }
    [data-testid="stMultiSelect"] span:has(button) span,
    [data-testid="stMultiSelect"] span:has(button) button,
    [data-baseweb="tag"] span {
        background-color: transparent !important;
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Reading model output…")
def cached_profile(model: str, pal: str, sza: str, var_id: str, planet: str, stamp: str) -> dict | None:
    del stamp
    profile = data.load_profile(model, pal, sza, var_id, planet)
    if profile is None:
        return None
    packed = dict(profile)
    for key in (
        "pressure_hpa",
        "value",
        "lat",
        "zonal",
        "zonal_pressure_hpa",
        "air_m3",
        "air_zonal",
        "density",
        "density_zonal",
    ):
        if packed.get(key) is not None:
            packed[key] = np.asarray(packed[key])
    return packed


def ozone_curve_stamp() -> str:
    """Changes when a curve input file is added or replaced.

    Streamlit keeps the ozone curve until this string changes, so a new
    Atmos run is included on the next refresh.
    """
    pieces = []
    for pal in data.CURVE_PALS:
        pieces.append(data.source_stamp("WACCM6", pal, "48.2", "O3"))
        for sza in ("45", "48.2", "60"):
            for model in ("Atmos", "Photochem", "Kasting", "VULCAN NCHO", "VULCAN SNCHOAr"):
                pieces.append(data.source_stamp(model, pal, sza, "O3"))
    return "\n".join(pieces)


@st.cache_data(show_spinner="Calculating ozone columns…")
def cached_ozone_curve(stamp: str) -> dict:
    del stamp
    return data.ozone_oxygen_curve()


def presented_profile(profile: dict | None, quantity: str) -> dict | None:
    """Return mixing ratio, or number density in molecules m^-3."""
    if profile is None or quantity != "density":
        return profile
    out = dict(profile)
    if profile.get("density") is not None:
        out["value"] = np.asarray(profile["density"], dtype=float)
        if profile.get("density_zonal") is not None:
            out["zonal"] = np.asarray(profile["density_zonal"], dtype=float)
        return out
    air = profile.get("air_m3")
    if air is None:
        return None
    out["value"] = np.asarray(profile["value"], dtype=float) * np.asarray(air, dtype=float)
    if profile.get("zonal") is not None:
        zonal = np.asarray(profile["zonal"], dtype=float)
        zonal_air = profile.get("air_zonal")
        if zonal_air is None:
            out["zonal"] = zonal * np.asarray(air, dtype=float)[:, None]
        else:
            out["zonal"] = zonal * np.asarray(zonal_air, dtype=float)
    return out


def display_meta(var_id: str, quantity: str) -> dict:
    meta = dict(data.VARIABLES[var_id])
    if quantity == "density" and var_id in data.CHEMICAL_IDS:
        meta["units"] = "molecules m⁻³"
        meta["label"] = f"{meta['label']} number density"
        meta["log"] = True
    return meta


# Fixed ranges follow the comparison figures in Early_Earth.py.
MIXING_X = {
    "O3": (1e-9, 1e-4),
    "O": (1e-12, 1e-1),
    "O2": (1e-9, 1.0),
    "CH4": (1e-9, 1e-4),
    "N2O": (1e-9, 1e-6),
    "OH": (2e-15, 1e-8),
    "H2O": (1e-7, 1e-4),
    "NOX": (1e-13, 1e-7),
    "HOX": (2e-13, 1e-7),
    "jo2_rate": (1e11, 6e12),
    "jo2": (1e-12, 1e-3),
    "jo2_a": (1e-12, 1e-3),
    "jo2_b": (1e-12, 1e-3),
    "jo3": (1e-12, 1e-2),
    "jo3_a": (1e-12, 1e-2),
    "jo3_b": (1e-12, 1e-2),
}
DENSITY_X = {
    "O3": (5e16, 6e18),
    "O": (1e9, 1e16),
    "O2": (1e16, 1e25),
    "CH4": (1e12, 1e21),
    "N2O": (1e12, 1e20),
    "OH": (1e8, 1e14),
    "H2O": (1e15, 1e24),
    "NOX": (1e10, 1e17),
    "HOX": (1e10, 1e17),
}
LINEAR_X = {
    "T": (145.0, 300.0),
    "U": (-60.0, 60.0),
    "V": (-10.0, 10.0),
    "CLDLIQ": (0.0, 8e-5),
    "CLDICE": (0.0, 5e-6),
}
ZONAL_FIELDS = ("U", "V", "CLDLIQ", "CLDICE")


def x_limits(var_id: str, quantity: str, planet: str = "earth") -> tuple[float, float] | None:
    if var_id == "jo2_int":
        return (5e15, 2e17)
    if planet == "proxima" and quantity != "density":
        if var_id == "O3":
            return (1e-10, 1e-4)
        if var_id == "jo2_rate":
            return (1e10, 5e12)
    if quantity == "density" and var_id in DENSITY_X:
        return DENSITY_X[var_id]
    if var_id in MIXING_X:
        return MIXING_X[var_id]
    return LINEAR_X.get(var_id)


def pressure_limits(var_id: str, quantity: str, planet: str = "earth") -> tuple[float, float]:
    """Surface pressure first, top-of-plot pressure second. Both in hPa.

    Windows follow the comparison figures in Early_Earth.py.
    """
    if var_id in ("CLDLIQ", "CLDICE"):
        return 1e3, 50.0
    if var_id == "jo2_int":
        return 1e3, 1e-5
    if planet == "proxima" and var_id in ("O3", "jo2_rate"):
        return 1e3, 1e-5
    if quantity == "density" and var_id in ("O3", "O"):
        return 1e3, 1e-1
    if var_id in ("NOX", "HOX", "H2O", "OH", "jo2", "jo2_a", "jo2_b", "jo2_rate"):
        return 1e3, 1e-3
    return 1e3, 1e-4


def axis_title(var_id: str, quantity: str) -> str:
    if var_id == "jo2_rate":
        return "O₂ photolysis rate [molecules m⁻³ s⁻¹]"
    if var_id == "jo2_int":
        return "Integrated O₂ photolysis [molecules m⁻² s⁻¹]"
    if var_id == "U":
        return "Zonal wind [m s⁻¹]"
    if var_id == "V":
        return "Meridional wind [m s⁻¹]"
    if var_id == "CLDLIQ":
        return "Cloud liquid [kg kg⁻¹]"
    if var_id == "CLDICE":
        return "Cloud ice [kg kg⁻¹]"
    if var_id == "T":
        return "Temperature [K]"
    base = data.VARIABLES[var_id]
    if quantity == "density":
        return f"{base['label']} number density [molecules m⁻³]"
    if base["kind"] in ("mixing", "nox", "hox"):
        return f"{base['label']} volume mixing ratio"
    if base["kind"] == "j":
        return f"{base['label']} [s⁻¹]"
    meta = display_meta(var_id, quantity)
    return f"{meta['label']} [{meta['units']}]"


def subplot_spacing(rows: int) -> dict:
    spacing = {"horizontal_spacing": 0.06}
    if rows > 1:
        spacing["vertical_spacing"] = 0.14
    return spacing


def apply_profile_axes(fig, var_id: str, quantity: str, planet: str = "earth") -> None:
    use_log = display_meta(var_id, quantity)["log"]
    limits = x_limits(var_id, quantity, planet)
    if limits is None:
        if use_log:
            fig.update_xaxes(type="log")
        return
    low, high = limits
    if use_log:
        fig.update_xaxes(type="log", range=[np.log10(low), np.log10(high)])
    else:
        fig.update_xaxes(range=[low, high])
    surface, top = pressure_limits(var_id, quantity, planet)
    fig.update_yaxes(type="log", range=[np.log10(surface), np.log10(top)])


def positive(values: np.ndarray, use_log: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    if use_log:
        array[array <= 0] = np.nan
    return array


def panel_shape(count: int) -> tuple[int, int]:
    if count <= 1:
        return 1, 1
    if count <= 3:
        return 1, count
    columns = 2 if count <= 4 else 4
    rows = int(np.ceil(count / columns))
    return rows, columns


def white_figure(fig) -> None:
    """White plot panel and white page behind it, with black text."""
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    fig.update_xaxes(
        gridcolor="#e6e6e6",
        zerolinecolor="#cccccc",
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )
    fig.update_yaxes(
        gridcolor="#e6e6e6",
        zerolinecolor="#cccccc",
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )


def style_pressure_axis(fig, rows: int, cols: int, var_id: str, quantity: str, planet: str = "earth") -> None:
    fig.update_layout(
        template="plotly_white",
        height=max(460, 300 * rows),
        legend=dict(orientation="h", yanchor="bottom", y=1.18, x=0),
        margin=dict(l=72, r=24, t=120, b=62),
    )
    white_figure(fig)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    x_title = axis_title(var_id, quantity)
    for col in range(1, cols + 1):
        fig.update_xaxes(title_text=x_title, row=rows, col=col)
    for row in range(1, rows + 1):
        fig.update_yaxes(title_text="Pressure [hPa]", row=row, col=1)
    apply_profile_axes(fig, var_id, quantity, planet)


def add_profile_line(fig, profile: dict, name: str, colour: str, use_log: bool, row: int, col: int, show_legend: bool) -> None:
    pressure = profile["pressure_hpa"]
    values = positive(profile["value"], use_log)
    fig.add_trace(
        go.Scatter(
            x=values,
            y=pressure,
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=show_legend,
            line=dict(color=colour, width=2.4),
            hovertemplate=f"{name}<br>%{{x:.3e}}<br>%{{y:.3g}} hPa<extra></extra>",
        ),
        row=row,
        col=col,
    )


def add_waccm_spread(fig, profile: dict, use_log: bool, row: int, col: int) -> None:
    zonal = profile.get("zonal")
    if zonal is None:
        return
    pressure = np.asarray(profile["pressure_hpa"], dtype=float)
    low = np.nanmin(zonal, axis=1)
    high = np.nanmax(zonal, axis=1)
    if use_log:
        low = positive(low, True)
        high = positive(high, True)
    mask = np.isfinite(pressure) & np.isfinite(low) & np.isfinite(high)
    if mask.sum() < 2:
        return
    pressure = pressure[mask]
    low = low[mask]
    high = high[mask]
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([low, high[::-1]]),
            y=np.concatenate([pressure, pressure[::-1]]),
            fill="toself",
            fillcolor="rgba(17, 17, 17, 0.13)",
            line=dict(width=0),
            hoverinfo="skip",
            name="WACCM6 latitude range",
            legendgroup="WACCM6 range",
            showlegend=(row == 1 and col == 1),
        ),
        row=row,
        col=col,
    )


def figure_compare(pals: list[str], models: list[str], sza: str, var_id: str, show_spread: bool, quantity: str, planet: str = "earth") -> tuple[go.Figure | None, list[str]]:
    meta = display_meta(var_id, quantity)
    use_log = meta["log"]
    rows, cols = panel_shape(len(pals))
    titles = [data.pal_label(pal) for pal in pals]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        shared_xaxes=False,
        shared_yaxes=True,
        **subplot_spacing(rows),
    )
    notes = []
    drawn = 0
    for index, pal in enumerate(pals):
        row = index // cols + 1
        col = index % cols + 1
        for model in models:
            if not meta["compare"] and model != "WACCM6":
                continue
            stamp = data.source_stamp(model, pal, sza, var_id, planet)
            if stamp.startswith("missing:"):
                if planet == "proxima" and model == "WACCM6":
                    notes.append(
                        f"WACCM6 has no condensed Proxima Centauri b chemistry file for {data.pal_label(pal)}."
                    )
                elif planet == "proxima":
                    notes.append(f"{model} has no Proxima Centauri b {data.pal_label(pal)} file.")
                else:
                    notes.append(f"{model} has no {data.pal_label(pal)} file at SZA {sza}°.")
                continue
            profile = presented_profile(cached_profile(model, pal, sza, var_id, planet, stamp), quantity)
            if profile is None:
                notes.append(f"{model} does not store {meta['label']} for {data.pal_label(pal)}.")
                continue
            if show_spread and model == "WACCM6":
                add_waccm_spread(fig, profile, use_log, row, col)
            add_profile_line(
                fig,
                profile,
                model,
                COLOURS[model],
                use_log,
                row,
                col,
                show_legend=(index == 0),
            )
            drawn += 1
    if drawn == 0:
        return None, notes
    style_pressure_axis(fig, rows, cols, var_id, quantity, planet)
    return fig, notes


def map_shape(count: int) -> tuple[int, int]:
    """Two columns so each latitude–pressure panel can carry its own colour bar."""
    if count <= 1:
        return 1, 1
    return int(np.ceil(count / 2)), 2


def map_colorscale(var_id: str) -> str:
    """Sequential fields follow the Early_Earth.py latitude–pressure figures."""
    if var_id in ("CLDLIQ", "CLDICE"):
        return "Blues_r"
    if var_id in ("U", "V", "T"):
        return "RdBu_r"
    if str(var_id).startswith("jo"):
        return "YlGnBu"
    return "YlOrRd"


def attach_colorbars(fig) -> None:
    """Place one colour bar against the right edge of each map panel."""
    for trace in fig.data:
        xname = getattr(trace, "xaxis", None) or "x"
        yname = getattr(trace, "yaxis", None) or "y"
        xkey = "xaxis" if xname == "x" else "xaxis" + xname[1:]
        ykey = "yaxis" if yname == "y" else "yaxis" + yname[1:]
        xdomain = fig.layout[xkey].domain
        ydomain = fig.layout[ykey].domain
        if xdomain is None or ydomain is None:
            continue
        title = ""
        kept = {}
        if trace.colorbar is not None:
            if trace.colorbar.title is not None:
                title = trace.colorbar.title.text or ""
            for key in ("tickformat", "tickmode", "tickvals", "ticktext"):
                value = getattr(trace.colorbar, key, None)
                if value is not None:
                    kept[key] = value
        colourbar = dict(
            title=dict(text=title, side="right", font=dict(size=12, color="black")),
            x=xdomain[1] + 0.012,
            xref="paper",
            y=(ydomain[0] + ydomain[1]) / 2.0,
            yref="paper",
            len=max((ydomain[1] - ydomain[0]) * 0.86, 0.12),
            lenmode="fraction",
            thickness=14,
            outlinewidth=0,
            tickfont=dict(size=11, color="black"),
        )
        colourbar.update(kept)
        trace.update(showscale=True, colorbar=colourbar)


def figure_maps(pals: list[str], sza: str, var_id: str, quantity: str) -> tuple[go.Figure | None, list[str]]:
    meta = display_meta(var_id, quantity)
    rows, cols = map_shape(len(pals))
    spacing = {"horizontal_spacing": 0.18}
    if rows > 1:
        spacing["vertical_spacing"] = 0.16
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[data.pal_label(pal) for pal in pals],
        **spacing,
    )
    notes = []
    drawn = 0
    diverging = bool(meta.get("diverging"))
    for index, pal in enumerate(pals):
        row = index // cols + 1
        col = index % cols + 1
        stamp = data.source_stamp("WACCM6", pal, sza, var_id, "earth")
        if stamp.startswith("missing:"):
            notes.append(f"No WACCM6 file for {data.pal_label(pal)}.")
            continue
        profile = presented_profile(cached_profile("WACCM6", pal, sza, var_id, "earth", stamp), quantity)
        if profile is None or profile.get("zonal") is None:
            notes.append(f"WACCM6 {data.pal_label(pal)} has no latitude–pressure field for {meta['label']}.")
            continue
        zonal = np.asarray(profile["zonal"], dtype=float)
        limits = x_limits(var_id, quantity)
        if meta["log"]:
            zonal = positive(zonal, True)
            plot_z = np.log10(zonal)
            colour_title = f"log₁₀ {axis_title(var_id, quantity)}"
            zmin = None if limits is None else np.log10(limits[0])
            zmax = None if limits is None else np.log10(limits[1])
        else:
            plot_z = zonal
            colour_title = axis_title(var_id, quantity)
            zmin = None if limits is None else limits[0]
            zmax = None if limits is None else limits[1]
        colourbar = dict(title=dict(text=colour_title))
        if var_id in ("CLDLIQ", "CLDICE"):
            colourbar["tickformat"] = ".1e"
        pressure = profile.get("zonal_pressure_hpa")
        if pressure is None:
            y_pressure = profile["pressure_hpa"]
        elif np.asarray(pressure).ndim == 2:
            y_pressure = np.nanmean(pressure, axis=1)
        else:
            y_pressure = pressure
        colours = map_colorscale(var_id)
        contour = dict(
            x=profile["lat"],
            y=y_pressure,
            z=plot_z,
            colorscale=colours,
            colorbar=colourbar,
            line=dict(width=0),
            contours=dict(coloring="fill"),
            showscale=True,
            hovertemplate="lat %{x:.1f}°<br>%{y:.3g} hPa<br>%{z:.3g}<extra></extra>",
        )
        if zmin is not None and zmax is not None and not diverging:
            contour["zmin"] = zmin
            contour["zmax"] = zmax
        if diverging:
            contour["zmid"] = 0
            if limits is not None:
                span = max(abs(limits[0]), abs(limits[1]))
                contour["zmin"] = -span
                contour["zmax"] = span
        fig.add_trace(go.Contour(**contour), row=row, col=col)
        drawn += 1
    if drawn == 0:
        return None, notes
    surface, top = pressure_limits(var_id, quantity)
    fig.update_yaxes(type="log", range=[np.log10(surface), np.log10(top)])
    fig.update_xaxes(range=[-90, 90], title_text="")
    fig.update_yaxes(title_text="")
    for col in range(1, cols + 1):
        fig.update_xaxes(title_text="Latitude [°]", row=rows, col=col)
    for row in range(1, rows + 1):
        fig.update_yaxes(title_text="Pressure [hPa]", row=row, col=1)
    fig.update_layout(
        template="plotly_white",
        height=max(480, 360 * rows),
        margin=dict(l=72, r=120, t=56, b=56),
    )
    attach_colorbars(fig)
    white_figure(fig)
    return fig, notes


def add_bands(fig, row: int, col: int, xmin: float, xmax: float, ymax: float) -> None:
    for x0, x1, label, fill, colour, label_x, height in BANDS:
        if x1 < xmin or x0 > xmax:
            continue
        fig.add_vrect(
            x0=max(x0, xmin),
            x1=min(x1, xmax),
            fillcolor=fill,
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )
        if not label:
            continue
        anchor = (x0 + x1) / 2 if label_x is None else label_x
        if not (xmin <= anchor <= xmax):
            continue
        fig.add_annotation(
            x=anchor,
            y=ymax * height,
            text=label,
            showarrow=False,
            font=dict(color=colour, size=13),
            row=row,
            col=col,
        )


def spectrum_panels(planet: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Visible panel, then infrared panel. Earth stops at 20 µm."""
    if planet == "proxima":
        return ((0.2, 1.5), (1.5, 12.0))
    return ((0.2, 1.5), (1.5, EARTH_WAVE_MAX))


def center_row_titles(fig, n_rows: int) -> None:
    """One oxygen-level title, centred across the two panels in that row."""
    for row in range(n_rows):
        left = fig.layout.annotations[row * 2]
        right = fig.layout.annotations[row * 2 + 1]
        left.update(x=(left.x + right.x) / 2, xanchor="center")
        right.update(text="")


def figure_spectra(planet: str, pals: list[str]) -> tuple[go.Figure | None, list[str]]:
    notes = []
    available = []
    for pal in pals:
        series = data.spectra_for(planet, pal)
        if not series:
            notes.append(f"No {planet_name(planet)} transmission file for {data.pal_label(pal)}.")
            continue
        available.append((pal, series))
    if not available:
        return None, notes
    n_pal = len(available)
    panels = spectrum_panels(planet)
    ymax = 80 if planet == "proxima" else 85
    spacing = {"horizontal_spacing": 0.06}
    if n_pal > 1:
        spacing["vertical_spacing"] = 0.12
    fig = make_subplots(
        rows=n_pal,
        cols=2,
        subplot_titles=[data.pal_label(pal) for pal, _series in available for _panel in panels],
        shared_yaxes=True,
        **spacing,
    )
    center_row_titles(fig, n_pal)
    spectrum_order = list(data.MODEL_ORDER) + ["VULCAN SNCHOAr"]
    has_sulfur = any("VULCAN SNCHOAr" in series for _pal, series in available)
    spectrum_names = {}
    if has_sulfur:
        spectrum_names["VULCAN"] = "VULCAN no S"
        spectrum_names["VULCAN SNCHOAr"] = "VULCAN S"
    legend_shown = set()
    for row, (_pal, series) in enumerate(available, start=1):
        for model in spectrum_order:
            if model not in series:
                continue
            wavelength, altitude = series[model]
            show = model not in legend_shown
            for col, (xmin, xmax) in enumerate(panels, start=1):
                mask = (wavelength >= xmin) & (wavelength <= xmax)
                colour = SPECTRUM_COLOURS[model]
                if has_sulfur and model == "VULCAN":
                    colour = SPECTRUM_NO_S
                line = dict(color=colour, width=2, dash="solid")
                fig.add_trace(
                    go.Scatter(
                        x=wavelength[mask],
                        y=altitude[mask],
                        mode="lines",
                        name=spectrum_names.get(model, model),
                        legendgroup=model,
                        showlegend=show and col == 1,
                        line=line,
                    ),
                    row=row,
                    col=col,
                )
            legend_shown.add(model)
        for col, (xmin, xmax) in enumerate(panels, start=1):
            add_bands(fig, row, col, xmin, xmax, ymax)
            fig.update_xaxes(range=[xmin, xmax], title_text="Wavelength [µm]", row=row, col=col)
        fig.update_yaxes(range=[0, ymax], title_text="Effective altitude [km]", row=row, col=1)
    title = "Proxima Centauri b transmission spectra" if planet == "proxima" else "Earth transmission spectra"
    fig.update_layout(
        template="plotly_white",
        height=max(440, 360 * n_pal),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        margin=dict(l=64, r=28, t=110, b=48),
    )
    white_figure(fig)
    return fig, notes


def fill_rgba(colour, alpha):
    # Remove leading '#' if present
    hex_code = colour.lstrip('#')
    
    red = int(hex_code[0:2], 16)
    green = int(hex_code[2:4], 16)
    blue = int(hex_code[4:6], 16)
    
    return f"rgba({red}, {green}, {blue}, {alpha})"


def add_column_band(
    fig: go.Figure,
    x,
    low,
    high,
    name: str,
    colour: str,
    alpha: float,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    """Shade a column range, breaking the shade where a point is missing."""
    x = np.asarray(x, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    mask = np.isfinite(x) & np.isfinite(low) & np.isfinite(high) & (x > 0)
    legend_drawn = False
    start = None
    for index, ok in enumerate(list(mask) + [False]):
        if ok and start is None:
            start = index
        elif not ok and start is not None:
            if index - start >= 2:
                xs = x[start:index]
                lo = low[start:index]
                hi = high[start:index]
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([xs, xs[::-1]]),
                        y=np.concatenate([lo, hi[::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba(colour, alpha),
                        line=dict(width=0, color=colour),
                        name=name,
                        legendgroup=name,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                if show_legend and not legend_drawn:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            line=dict(color=colour, width=12),
                            name=name,
                            legendgroup=name,
                            showlegend=True,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                legend_drawn = True
            start = None


def add_column_line(
    fig: go.Figure,
    x,
    y,
    name: str,
    colour: str,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() < 2:
        return
    fig.add_trace(
        go.Scatter(
            x=x[mask],
            y=y[mask],
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=show_legend,
            line=dict(color=colour, width=2.2),
        ),
        row=row,
        col=col,
    )


def figure_ozone_column(curve: dict) -> go.Figure:
    """Paper O2–O3 figure: overview on top, then one panel for each 1D model."""
    # Same colours as the paper panels. VULCAN stays the two networks.
    ozone_colours = {
        "WACCM6": "#000000",
        "Total 1D model range": "#00bcd4",
        "Kasting 1D range": "#008080",
        "Photochem range": "#0000ff",
        "Atmos range": "#ff8c00",
        "VULCAN NCHO": "#6b21a8",
        "VULCAN SNCHOAr": "#ff00ff",
    }
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
        vertical_spacing=0.14,
        horizontal_spacing=0.07,
    )
    x = curve["x"]
    panels = (
        (1, 1, "overview"),
        (2, 1, "Kasting 1D range"),
        (2, 2, "Photochem range"),
        (3, 1, "Atmos range"),
        (3, 2, "VULCAN"),
    )
    x_title = "Oxygen mixing ratio [PAL]"
    y_title = "O₃ column [DU]"
    for row, col, kind in panels:
        add_column_band(
            fig, x, curve["waccm_min"], curve["waccm_max"],
            "WACCM6 (Cooke et al. 2022)", ozone_colours["WACCM6"], 0.18,
            row, col, show_legend=(row == 1 and col == 1),
        )
        add_column_line(
            fig, x, curve["waccm_mean"],
            "WACCM6 (Cooke et al. 2022)", ozone_colours["WACCM6"],
            row, col, show_legend=False,
        )
        if kind == "overview":
            add_column_band(
                fig, x, curve["envelope_low"], curve["envelope_high"],
                "Total 1D model range", ozone_colours["Total 1D model range"], 0.4,
                row, col, show_legend=True,
            )
        elif kind == "VULCAN":
            for name, alpha in (("VULCAN NCHO", 0.45), ("VULCAN SNCHOAr", 0.75)):
                band = curve["bands"][name]
                add_column_band(
                    fig, x, band["low"], band["high"],
                    name, ozone_colours[name], alpha, row, col, show_legend=True,
                )
        else:
            model = kind.split()[0]
            band = curve["bands"][model]
            add_column_band(
                fig, x, band["low"], band["high"],
                kind, ozone_colours[kind], 0.4, row, col, show_legend=True,
            )
        fig.update_xaxes(
            type="log",
            range=[np.log10(1.0e-3), np.log10(1.5)],
            tickvals=[0.001, 0.01, 0.1, 1],
            ticktext=["10⁻³", "10⁻²", "10⁻¹", "1"],
            row=row,
            col=col,
        )
        fig.update_yaxes(range=[0, 380], row=row, col=col)
    fig.update_xaxes(title_text=x_title, row=1, col=1)
    fig.update_xaxes(title_text=x_title, row=3, col=1)
    fig.update_xaxes(title_text=x_title, row=3, col=2)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_yaxes(title_text=y_title, row=3, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=3, col=2)
    # The spanned top panel means a title list is assigned to the wrong axes.
    # Place each name on the panel that actually holds that model.
    panel_titles = {
        ("xaxis2", "yaxis2"): "Kasting",
        ("xaxis3", "yaxis3"): "Photochem",
        ("xaxis4", "yaxis4"): "Atmos",
        ("xaxis5", "yaxis5"): "VULCAN",
    }
    for (x_name, y_name), title in panel_titles.items():
        x_domain = fig.layout[x_name].domain
        y_domain = fig.layout[y_name].domain
        fig.add_annotation(
            text=title,
            x=(x_domain[0] + x_domain[1]) / 2.0,
            y=y_domain[1],
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            yshift=6,
            showarrow=False,
            font=dict(size=16, color="black"),
        )
    fig.update_layout(
        template="plotly_white",
        height=1120,
        title=dict(
            text="O₂–O₃ curve between 1D models and WACCM6",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=18, color="black"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            x=0.5,
            xanchor="center",
            font=dict(size=14, color="black"),
            bgcolor="white",
            itemsizing="constant",
            tracegroupgap=18,
        ),
        margin=dict(l=78, r=36, t=210, b=56),
    )
    white_figure(fig)
    return fig


def planet_name(planet: str) -> str:
    if planet == "proxima":
        return "Proxima Centauri b"
    return "Earth"


def main() -> None:
    st.title("Simulations of the Evolving Ozone Layer")
    st.caption(
        "Interactive supplement for Kumar & Cooke 2026, Royal Society Open Science: "
        "Implications for the Early Earth, Exoplanets, and the Search For Life. "
        "Standard oxygen levels are 100%, 10%, 1%, and 0.1% PAL. "
        "150%, 50%, 5%, and 0.5% PAL stay available in the oxygen menu."
    )

    with st.sidebar:
        st.header("Explorer")
        style = st.selectbox(
            "Plot style",
            [STYLE_COMPARE, STYLE_OZONE, STYLE_MAP, STYLE_SPECTRUM],
        )
        planet = "earth"
        if style == STYLE_SPECTRUM:
            planet = st.radio(
                "Planet",
                ["proxima", "earth"],
                format_func=planet_name,
                key="spectrum_planet",
            )
        elif style == STYLE_COMPARE:
            planet = st.radio(
                "Planet",
                ["earth", "proxima"],
                format_func=planet_name,
                horizontal=True,
                key="compare_planet",
            )
        variable_labels = {key: data.VARIABLES[key]["label"] for key in data.VARIABLES}
        if style == STYLE_OZONE:
            var_id = "O3"
            pals = list(data.CURVE_PALS)
        else:
            var_id = st.selectbox(
                "Variable",
                list(variable_labels),
                format_func=lambda key: variable_labels[key],
                index=0,
            )
            if planet == "proxima" and style == STYLE_COMPARE:
                oxygen_choices = data.PROXIMA_WACCM_PALS
                oxygen_default = list(data.PROXIMA_WACCM_PALS)
            elif planet == "proxima":
                oxygen_choices = data.PROXIMA_PALS
                oxygen_default = list(data.PROXIMA_PALS)
            else:
                oxygen_choices = data.PAL_ORDER
                oxygen_default = list(data.STANDARD_PALS)
            pals = st.multiselect(
                "Oxygen",
                oxygen_choices,
                default=oxygen_default,
                format_func=data.pal_label,
                key=f"oxygen_{style}_{planet}",
            )
        meta = data.VARIABLES[var_id]
        sza = "48.2"
        models = list(data.PROFILE_MODELS)
        show_spread = False
        quantity = "mixing"
        if var_id in data.CHEMICAL_IDS and style in (STYLE_COMPARE, STYLE_MAP):
            quantity = st.radio(
                "Quantity",
                ["mixing", "density"],
                format_func=lambda key: "Mixing ratio" if key == "mixing" else "Number density",
                horizontal=True,
            )
        if style == STYLE_COMPARE:
            if planet == "earth" and meta["compare"]:
                sza = st.selectbox("1D solar zenith angle", data.SZA_CHOICES, index=0)
                models = st.multiselect(
                    "Models",
                    data.PROFILE_MODELS,
                    default=data.PROFILE_MODELS,
                    key="compare_models_earth",
                )
            elif planet == "proxima":
                models = ["WACCM6"]
            show_spread = st.checkbox("WACCM6 range across latitude", value=True)
        if planet == "proxima" and style == STYLE_COMPARE:
            st.caption(
                "Proxima Centauri b chemistry is the WACCM6 zonal mean at 100% and 1% PAL."
            )
        elif planet == "proxima":
            st.caption("Proxima Centauri b transmission spectra use the saved PSG files.")
        else:
            st.caption(
                "WACCM6 curves use the zonal-mean files in WACCM6. "
                "10% to 0.1% PAL use the corrected upper-boundary runs."
            )

    if not pals:
        st.info("Choose at least one oxygen level.")
        return

    if style == STYLE_OZONE:
        curve = cached_ozone_curve(ozone_curve_stamp())
        figure = figure_ozone_column(curve)
        notes = curve["notes"]
        st.caption(
            "Ozone column against oxygen, as in the paper figure. "
            "The black line is the WACCM6 Gaussian-weighted mean and the grey band is "
            "the minimum to maximum of the zonal mean over latitude. "
            "Kasting, Photochem, and Atmos are shaded from 45° to 60°. "
            "VULCAN NCHO is the darker purple band from 48.2° to 60°, and "
            "VULCAN SNCHOAr is the lighter pink band from 45° to 60°. "
            "The 50% PAL SNCHOAr run at 60° is omitted: its temperature stays "
            "at 287 K through the stratosphere, so it is not a WACCM-temperature case. "
            "Kasting at 50% PAL uses the same output for 45° and 60°, so that shade closes. "
            "Seasonal wind and circulation figures are not included."
        )
    elif style == STYLE_SPECTRUM:
        figure, notes = figure_spectra(planet, pals)
        if planet == "earth":
            st.caption(
                "Earth spectra are the updated 60° PSG runs for WACCM6, Atmos, Photochem, "
                "VULCAN, and Kasting at 100%, 10%, 1%, and 0.1% PAL. "
                "Lines match the paper: solid black WACCM6, blue Photochem, "
                "dark orange Atmos, teal Kasting, and magenta VULCAN. "
                "The 150% PAL panel is the earlier VULCAN spectrum. "
                "The infrared panel goes out to 20 µm."
            )
        else:
            st.caption(
                "Proxima Centauri b spectra for 100%, 10%, 1%, and 0.1% PAL. "
                "Lines match the paper: solid black WACCM6, blue Photochem, dark orange Atmos, "
                "dark purple VULCAN no S, and magenta VULCAN S. "
                "The 0.1% PAL sulfur run is not shown: that file still has 100% PAL oxygen. "
                "The paper figure is the 1% PAL case. Shading marks O₃ and O₂ bands."
            )
    elif var_id in ZONAL_FIELDS and planet != "proxima":
        shown = display_meta(var_id, quantity)
        figure, notes = figure_maps(pals, sza, var_id, quantity)
        cloud = var_id in ("CLDLIQ", "CLDICE")
        span = "Pressure runs from 1000 hPa to 50 hPa." if cloud else "Pressure runs from 1000 hPa to 10⁻⁴ hPa."
        st.caption(
            f"WACCM6 zonal mean of {shown['label']}. "
            f"Longitude is averaged. {span}"
            + (" The colour scale is linear, in kg kg⁻¹." if cloud else "")
        )
    elif style == STYLE_COMPARE:
        if not models:
            st.info("Choose at least one model.")
            return
        figure, notes = figure_compare(pals, models, sza, var_id, show_spread, quantity, planet)
        st.caption(profile_caption(var_id, sza, quantity, planet))
    elif style == STYLE_MAP:
        shown = display_meta(var_id, quantity)
        figure, notes = figure_maps(pals, sza, var_id, quantity)
        st.caption(f"WACCM6 zonal mean of {shown['label']}. {family_caption(var_id, quantity)}")

    if figure is None:
        st.warning("Nothing to plot for this selection.")
    else:
        st.plotly_chart(figure, width="stretch", theme=None)

    unique_notes = []
    for note in notes:
        if note not in unique_notes:
            unique_notes.append(note)
    if unique_notes:
        with st.expander(f"Missing files ({len(unique_notes)})"):
            for note in unique_notes:
                st.write(note)

    with st.expander("Where the curves come from"):
        st.markdown(
            """
            1D profiles are the present-Sun, WACCM6 temperature-profile experiments
            in `Atmos/`, `Photochem/`, and `Kasting_1D_model/`. VULCAN Earth
            profiles are the lower-boundary, WACCM temperature-profile runs.
            NCHO is the darker purple curve and SNCHOAr is the lighter pink
            curve. The app reads the condensed chemistry in `VULCAN/output`
            (`*.npz`: every species, both mixing ratio and number density,
            temperature, pressure, and every photolysis frequency). Full
            `.vul` files in `~/VULCAN/output` are used when a condensed file
            is not there yet.

            WACCM6 profiles come from the zonal-mean files in `WACCM6/`
            (0.1% to 150% PAL). Longitude is already averaged. The black line is the
            Gaussian-weighted global mean. The shaded band is the minimum to
            maximum of that zonal mean across latitude. Latitude–pressure maps use the same
            files. 10% to 0.1% PAL are the corrected upper-boundary runs.

            NOₓ is N + NO + NO₂. HOₓ is H + OH + HO₂ + 2 H₂O₂. Those are the
            WACCM6 family definitions. Photolysis curves are frequencies J in
            s⁻¹. J(O₃) branch a is O₃ → O₂ + O(¹D) (WACCM6 `jo3_a`, Kasting
            `PO3D`, VULCAN branch 2). Branch b is O₃ → O₂ + O(³P). J(O₂) branch
            a is WACCM6 `jo2_a`, Kasting `PO2D`, and VULCAN O₂ branch 2.
            Photochem output does not store J. Its O₂ photolysis rate is the
            odd-oxygen production from O₂ + hv → O + O and O₂ + hv → O + O(¹D),
            evaluated on the saved atmosphere as in Early_Earth.py. The 48.2°
            curve uses that script’s settings file, whose solar zenith angle
            is 60°. Atmos stores O₂ photolysis only, in `out.O2prates`. The
            O₂ photolysis rate is the odd-oxygen production in molecules m⁻³ s⁻¹.
            WACCM6, Kasting, and Atmos use 2 J(O₂) n(O₂). VULCAN uses
            2 J(O₂) n(O₂) × 3/8, with J taken from the total O₂ branch
            (`J_sp` branch 0) and n from the saved number density.

            Integrated O₂ photolysis is the column above each level, in
            molecules m⁻² s⁻¹, using the same integrals as Early_Earth.py.
            WACCM6 sums 2 J(O₂) times the O₂ column in each hybrid layer from
            the top downward, with g = 9.81. Photochem multiplies the local
            rate by 1000 m. Atmos and Kasting integrate along the saved
            altitude. VULCAN uses the saved layer thickness.

            Proxima Centauri b chemistry in the comparison plot is the WACCM6
            zonal mean at 100% and 1% PAL, from the condensed files in
            `WACCM6/proxima`. Transmission spectra are separate and include
            WACCM6, Atmos, Photochem, and both VULCAN networks.

            Zonal wind, meridional wind, cloud liquid, and cloud ice are shown
            as WACCM6 zonal means: longitude is averaged, and the plot is
            latitude against pressure.

            Number density is molecules m⁻³. WACCM6, Atmos, and Photochem use
            mixing ratio × P / (kT), with the same k as the Early_Earth.py
            density figures. Kasting uses DEN, converted from cm⁻³. VULCAN uses
            the saved number density, converted from cm⁻³. O₃ number density
            uses the same axis window as that figure: 5×10¹⁶ to 6×10¹⁸ m⁻³
            and 1000 to 0.1 hPa.
            """
        )


def profile_caption(var_id: str, sza: str, quantity: str, planet: str = "earth") -> str:
    meta = display_meta(var_id, quantity)
    if planet == "proxima":
        text = (
            f"Proxima Centauri b WACCM6 {axis_title(var_id, quantity)} "
            "at 100% and 1% PAL."
        )
    else:
        text = f"{axis_title(var_id, quantity)} at a 1D solar zenith angle of {sza}°."
    if not meta["compare"]:
        text += " U, V, cloud liquid, and cloud ice are WACCM6 fields."
    extra = family_caption(var_id, quantity, planet)
    if extra:
        text += " " + extra
    return text


def family_caption(var_id: str, quantity: str = "mixing", planet: str = "earth") -> str:
    parts = []
    if var_id == "NOX":
        parts.append("NOₓ is N + NO + NO₂.")
    elif var_id == "HOX":
        parts.append("HOₓ is H + OH + HO₂ + 2 H₂O₂.")
    elif var_id.startswith("jo") and var_id not in ("jo2_rate", "jo2_int"):
        parts.append("J is the photolysis frequency in s⁻¹.")
    if var_id == "jo2_rate":
        parts.append(
            "WACCM6, Kasting, and Atmos use 2 J(O₂) n(O₂), with J(O₂) = jo2_a + jo2_b. "
            "VULCAN uses 2 J(O₂ branch 0) n(O₂) × 3/8. "
            "Photochem is the odd-oxygen production from the two O₂ + hv branches, "
            "computed on the saved atmosphere."
        )
        if planet == "proxima":
            parts.append("The axis runs from 10¹⁰ to 5×10¹², and pressure from 1000 to 10⁻⁵ hPa.")
    if var_id == "jo2_int":
        parts.append(
            "The column is integrated from the top of the atmosphere, in molecules m⁻² s⁻¹. "
            "The axis runs from 5×10¹⁵ to 2×10¹⁷, and pressure from 1000 to 10⁻⁵ hPa."
        )
    if quantity == "mixing" and var_id in data.CHEMICAL_IDS:
        if var_id == "OH":
            parts.append("The axis runs from 2×10⁻¹⁵ to 10⁻⁸, and pressure from 1000 to 0.001 hPa.")
        elif var_id == "NOX":
            parts.append("The axis runs from 10⁻¹³ to 10⁻⁷, and pressure from 1000 to 0.001 hPa.")
        elif var_id == "HOX":
            parts.append("The axis runs from 2×10⁻¹³ to 10⁻⁷, and pressure from 1000 to 0.001 hPa.")
        elif var_id == "H2O":
            parts.append("The axis runs from 10⁻⁷ to 10⁻⁴, and pressure from 1000 to 0.001 hPa.")
        elif var_id == "O":
            parts.append("This axis extends below 10⁻⁹.")
        elif var_id == "O3" and planet == "proxima":
            parts.append("The axis runs from 10⁻¹⁰ to 10⁻⁴, and pressure from 1000 to 10⁻⁵ hPa.")
        elif var_id == "O3":
            parts.append("The axis runs from 10⁻⁹ to 10⁻⁴.")
        else:
            parts.append("The mixing-ratio axis does not go below 10⁻⁹.")
    if quantity == "density" and var_id == "O3":
        parts.append("The axis runs from 5×10¹⁶ to 6×10¹⁸ molecules m⁻³, from 1000 to 0.1 hPa.")
    elif quantity == "density" and var_id == "O":
        parts.append("The axis runs from 10⁹ to 10¹⁶ molecules m⁻³, from 1000 to 0.1 hPa.")
    elif quantity == "density" and var_id in data.CHEMICAL_IDS:
        parts.append("Number density is in molecules m⁻³.")
    return " ".join(parts)


if __name__ == "__main__":
    main()
else:
    main()
