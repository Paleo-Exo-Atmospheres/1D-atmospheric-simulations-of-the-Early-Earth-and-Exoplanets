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
    "VULCAN": "#c026d3",
    "Kasting": "#0f766e",
}

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
def cached_profile(model: str, pal: str, sza: str, var_id: str, _stamp: str) -> dict | None:
    del _stamp
    profile = data.load_profile(model, pal, sza, var_id)
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
# Mixing-ratio axes stay at or above 1e-9, except O and OH.
MIXING_X = {
    "O3": (1e-9, 1e-4),
    "O": (1e-12, 1e-1),
    "O2": (1e-9, 1.0),
    "CH4": (1e-9, 1e-4),
    "N2O": (1e-9, 1e-6),
    "OH": (2e-15, 1e-8),
    "H2O": (1e-9, 1e-1),
    "NOX": (1e-12, 1e-7),
    "HOX": (1e-12, 1e-7),
    "jo2_rate": (1e11, 6e12),
    "jo2": (1e-12, 1e-2),
    "jo2_a": (1e-12, 1e-2),
    "jo2_b": (1e-12, 1e-2),
    "jo3": (1e-12, 1e-2),
    "jo3_a": (1e-12, 1e-2),
    "jo3_b": (1e-12, 1e-2),
    "CLDLIQ": (1e-8, 1e-3),
    "CLDICE": (1e-8, 1e-3),
}
DENSITY_X = {
    "O3": (5e16, 6e18),
    "O": (1e10, 1e18),
    "O2": (1e16, 1e25),
    "CH4": (1e12, 1e21),
    "N2O": (1e12, 1e20),
    "OH": (1e8, 1e14),
    "H2O": (1e15, 1e24),
    "NOX": (1e10, 1e17),
    "HOX": (1e10, 1e17),
}
LINEAR_X = {
    "T": (150.0, 310.0),
    "U": (-50.0, 50.0),
    "V": (-10.0, 10.0),
}
ZONAL_FIELDS = ("U", "V", "CLDLIQ", "CLDICE")


def x_limits(var_id: str, quantity: str) -> tuple[float, float] | None:
    if quantity == "density" and var_id in DENSITY_X:
        return DENSITY_X[var_id]
    if var_id in MIXING_X:
        return MIXING_X[var_id]
    return LINEAR_X.get(var_id)


def pressure_limits(var_id: str, quantity: str) -> tuple[float, float]:
    """Surface pressure first, top-of-plot pressure second. Both in hPa."""
    del quantity
    if var_id in ("CLDLIQ", "CLDICE"):
        return 1e3, 50.0
    return 1e3, 1e-4


def axis_title(var_id: str, quantity: str) -> str:
    if var_id == "jo2_rate":
        return "O₂ photolysis rate [molecules m⁻³ s⁻¹]"
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


def apply_profile_axes(fig, var_id: str, quantity: str) -> None:
    use_log = display_meta(var_id, quantity)["log"]
    limits = x_limits(var_id, quantity)
    if limits is None:
        if use_log:
            fig.update_xaxes(type="log")
        return
    low, high = limits
    if use_log:
        fig.update_xaxes(type="log", range=[np.log10(low), np.log10(high)])
    else:
        fig.update_xaxes(range=[low, high])
    surface, top = pressure_limits(var_id, quantity)
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


def style_pressure_axis(fig, rows: int, cols: int, var_id: str, quantity: str) -> None:
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
    apply_profile_axes(fig, var_id, quantity)


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


def figure_compare(pals: list[str], models: list[str], sza: str, var_id: str, show_spread: bool, quantity: str) -> tuple[go.Figure | None, list[str]]:
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
            stamp = data.source_stamp(model, pal, sza, var_id)
            if stamp.startswith("missing:"):
                notes.append(f"{model} has no {data.pal_label(pal)} file at SZA {sza}°.")
                continue
            profile = presented_profile(cached_profile(model, pal, sza, var_id, stamp), quantity)
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
    style_pressure_axis(fig, rows, cols, var_id, quantity)
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
        if trace.colorbar is not None and trace.colorbar.title is not None:
            title = trace.colorbar.title.text or ""
        trace.update(
            showscale=True,
            colorbar=dict(
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
            ),
        )


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
        stamp = data.source_stamp("WACCM6", pal, sza, var_id)
        if stamp.startswith("missing:"):
            notes.append(f"No WACCM6 file for {data.pal_label(pal)}.")
            continue
        profile = presented_profile(cached_profile("WACCM6", pal, sza, var_id, stamp), quantity)
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
            colorbar=dict(title=dict(text=colour_title)),
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
    for row, (_pal, series) in enumerate(available, start=1):
        for model in data.MODEL_ORDER:
            if model not in series:
                continue
            wavelength, altitude = series[model]
            for col, (xmin, xmax) in enumerate(panels, start=1):
                mask = (wavelength >= xmin) & (wavelength <= xmax)
                fig.add_trace(
                    go.Scatter(
                        x=wavelength[mask],
                        y=altitude[mask],
                        mode="lines",
                        name=model,
                        legendgroup=model,
                        showlegend=(row == 1 and col == 1),
                        line=dict(color=COLOURS[model], width=2.2),
                    ),
                    row=row,
                    col=col,
                )
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
            [STYLE_COMPARE, STYLE_MAP, STYLE_SPECTRUM],
        )
        variable_labels = {key: data.VARIABLES[key]["label"] for key in data.VARIABLES}
        var_id = st.selectbox(
            "Variable",
            list(variable_labels),
            format_func=lambda key: variable_labels[key],
            index=0,
        )
        pal_options = data.PAL_ORDER
        pals = st.multiselect(
            "Oxygen",
            pal_options,
            default=data.STANDARD_PALS,
            format_func=data.pal_label,
        )
        meta = data.VARIABLES[var_id]
        sza = "48.2"
        models = list(data.MODEL_ORDER)
        show_spread = False
        planet = "proxima"
        quantity = "mixing"
        if var_id in data.CHEMICAL_IDS and style != STYLE_SPECTRUM:
            quantity = st.radio(
                "Quantity",
                ["mixing", "density"],
                format_func=lambda key: "Mixing ratio" if key == "mixing" else "Number density",
                horizontal=True,
            )
        if style != STYLE_SPECTRUM:
            if meta["compare"] and style == STYLE_COMPARE:
                sza = st.selectbox("1D solar zenith angle", data.SZA_CHOICES, index=0)
                models = st.multiselect("Models", data.MODEL_ORDER, default=data.MODEL_ORDER)
            if style == STYLE_COMPARE:
                show_spread = st.checkbox("WACCM6 range across latitude", value=True)
        else:
            planet = st.radio(
                "Planet",
                ["proxima", "earth"],
                format_func=planet_name,
            )
        st.caption(
            "WACCM6 curves use the compressed subsets in Plots/waccm. "
            "150% PAL is not in that set."
        )

    if not pals:
        st.info("Choose at least one oxygen level.")
        return

    if style == STYLE_SPECTRUM:
        figure, notes = figure_spectra(planet, pals)
        if planet == "earth":
            st.caption(
                "Earth spectra use the same two panels as Proxima Centauri b, out to 20 µm. "
                "VULCAN has 150% PAL and no 100% PAL Earth spectrum. "
                "This set has no Earth WACCM6 transmission spectrum."
            )
        else:
            st.caption(
                "Proxima Centauri b spectra for 100%, 10%, 1%, and 0.1% PAL. "
                "The paper figure is the 1% PAL case. Shading marks O₃ and O₂ bands."
            )
    elif var_id in ZONAL_FIELDS:
        shown = display_meta(var_id, quantity)
        figure, notes = figure_maps(pals, sza, var_id, quantity)
        cloud = var_id in ("CLDLIQ", "CLDICE")
        span = "Pressure runs from 1000 hPa to 50 hPa." if cloud else "Pressure runs from 1000 hPa to 10⁻⁴ hPa."
        st.caption(
            f"WACCM6 zonal mean of {shown['label']}. "
            f"Longitude is averaged. {span}"
        )
    elif style == STYLE_COMPARE:
        if not models:
            st.info("Choose at least one model.")
            return
        figure, notes = figure_compare(pals, models, sza, var_id, show_spread, quantity)
        st.caption(profile_caption(var_id, sza, quantity))
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
            in `Atmos/`, `Photochem/`, and `Kasting_1D_model/`. VULCAN `.vul` files
            are read from `VULCAN/output` in this repository or from
            `~/VULCAN/output`.

            WACCM6 profiles come from the compressed files in `Plots/waccm`
            (0.1% to 100% PAL, plus 0.5%, 5%, and 50%). Each file still has
            longitude, and the app averages it. The black line is the
            Gaussian-weighted global mean. The shaded band is the minimum to
            maximum of that zonal mean. Latitude–pressure maps use the same
            zonal mean. 150% PAL was not included in the GitHub subset.

            NOₓ is N + NO + NO₂. HOₓ is H + OH + HO₂ + 2 H₂O₂. Those are the
            WACCM6 family definitions. Photolysis curves are frequencies J in
            s⁻¹. J(O₃) branch a is O₃ → O₂ + O(¹D) (WACCM6 `jo3_a`, Kasting
            `PO3D`, VULCAN branch 2). Branch b is O₃ → O₂ + O(³P). J(O₂) branch
            a is WACCM6 `jo2_a`, Kasting `PO2D`, and VULCAN O₂ branch 2.
            Photochem files in this archive do not contain J values. Atmos
            stores O₂ photolysis only, in `out.O2prates`. The O₂ photolysis
            rate is 2 J(O₂) n(O₂), in molecules m⁻³ s⁻¹. That is the
            odd-oxygen production rate from Early_Earth.py.

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


def profile_caption(var_id: str, sza: str, quantity: str) -> str:
    meta = display_meta(var_id, quantity)
    text = f"{axis_title(var_id, quantity)} at a 1D solar zenith angle of {sza}°."
    if not meta["compare"]:
        text += " U, V, cloud liquid, and cloud ice are WACCM6 fields."
    extra = family_caption(var_id, quantity)
    if extra:
        text += " " + extra
    return text


def family_caption(var_id: str, quantity: str = "mixing") -> str:
    parts = []
    if var_id == "NOX":
        parts.append("NOₓ is N + NO + NO₂.")
    elif var_id == "HOX":
        parts.append("HOₓ is H + OH + HO₂ + 2 H₂O₂.")
    elif var_id.startswith("jo"):
        parts.append("J is the photolysis frequency in s⁻¹.")
    if var_id == "jo2_rate":
        parts.append("2 J(O₂) n(O₂), with J(O₂) = jo2_a + jo2_b.")
    if quantity == "mixing" and var_id in data.CHEMICAL_IDS:
        if var_id in ("O", "OH"):
            parts.append("This axis extends below 10⁻⁹.")
        elif var_id in ("NOX", "HOX"):
            parts.append("This axis extends down to 10⁻¹².")
        else:
            parts.append("The mixing-ratio axis does not go below 10⁻⁹.")
    if quantity == "density" and var_id in data.CHEMICAL_IDS:
        parts.append("Number density is in molecules m⁻³.")
    return " ".join(parts)


if __name__ == "__main__":
    main()
else:
    main()
