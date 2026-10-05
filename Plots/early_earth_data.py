"""Load Early Earth 1D profiles, WACCM6 fields, and PSG transmission spectra.

The Streamlit app imports this module. Paths are relative to the
1D-Simulations-of-the-Early-Earth repository. WACCM6 fields are the compressed
subsets in ``Plots/waccm``. VULCAN ``.vul`` files are read from this repository
or from ``~/VULCAN/output``.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent
REPO_DIR = PLOTS_DIR.parent
SPECTRA_DIR = PLOTS_DIR / "spectra"
WACCM_SUBSET_DIR = PLOTS_DIR / "waccm"

PAL_ORDER = ["150", "100", "50", "10", "5", "1", "0.5", "0.1"]
STANDARD_PALS = ["100", "10", "1", "0.1"]
EXTRA_PALS = ["150", "50", "5", "0.5"]
SZA_CHOICES = ["48.2", "45", "60"]

MODEL_ORDER = ["WACCM6", "Atmos", "Photochem", "VULCAN", "Kasting"]
PROFILE_MODELS = [
    "WACCM6",
    "Atmos",
    "Photochem",
    "VULCAN NCHO",
    "VULCAN SNCHOAr",
    "Kasting",
]

# WACCM6 family definitions, used for every model so the curves are the same quantity.
# NOX long_name: "nox (N+NO+NO2)"
# HOX long_name: "HOx (H+OH+HO2+2H2O2)"
NOX_SPECIES = ("N", "NO", "NO2")
HOX_SPECIES = ("H", "OH", "HO2", "H2O2")

# jo3_a is O3 -> O2 + O(1D). jo3_b is O3 -> O2 + O(3P).
# Kasting PO3D / PO3 follow the same split (PO3 is only in outchem.dat).
# VULCAN J_sp branch 0 is the sum. Branch 1 is the O(3P) channel and branch 2
# is the O(1D) channel in the Earth PAL network used for this paper: at the
# surface, branch 1 is the larger tropospheric rate.
VARIABLES = {
    "O3": {
        "label": "O₃",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "mixing",
    },
    "O": {
        "label": "O",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "mixing",
    },
    "O2": {
        "label": "O₂",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "mixing",
    },
    "CH4": {
        "label": "CH₄",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "mixing",
    },
    "N2O": {
        "label": "N₂O",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "mixing",
    },
    "OH": {
        "label": "OH",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "mixing",
    },
    "H2O": {
        "label": "H₂O",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "mixing",
    },
    "NOX": {
        "label": "NOₓ (N + NO + NO₂)",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "nox",
    },
    "HOX": {
        "label": "HOₓ (H + OH + HO₂ + 2 H₂O₂)",
        "units": "mol mol⁻¹",
        "log": True,
        "compare": True,
        "kind": "hox",
    },
    "T": {
        "label": "Temperature",
        "units": "K",
        "log": False,
        "compare": True,
        "kind": "temperature",
    },
    "jo2": {
        "label": "J(O₂) total",
        "units": "s⁻¹",
        "log": True,
        "compare": True,
        "kind": "j",
    },
    "jo2_a": {
        "label": "J(O₂) branch a",
        "units": "s⁻¹",
        "log": True,
        "compare": True,
        "kind": "j",
    },
    "jo2_b": {
        "label": "J(O₂) branch b",
        "units": "s⁻¹",
        "log": True,
        "compare": True,
        "kind": "j",
    },
    "jo2_rate": {
        "label": "O₂ photolysis rate",
        "units": "molecules m⁻³ s⁻¹",
        "log": True,
        "compare": True,
        "kind": "rate",
    },
    "jo3": {
        "label": "J(O₃) total",
        "units": "s⁻¹",
        "log": True,
        "compare": True,
        "kind": "j",
    },
    "jo3_a": {
        "label": "J(O₃) → O(¹D)",
        "units": "s⁻¹",
        "log": True,
        "compare": True,
        "kind": "j",
    },
    "jo3_b": {
        "label": "J(O₃) → O(³P)",
        "units": "s⁻¹",
        "log": True,
        "compare": True,
        "kind": "j",
    },
    "U": {
        "label": "Zonal wind U",
        "units": "m s⁻¹",
        "log": False,
        "compare": False,
        "kind": "wind",
        "diverging": True,
    },
    "V": {
        "label": "Meridional wind V",
        "units": "m s⁻¹",
        "log": False,
        "compare": False,
        "kind": "wind",
        "diverging": True,
    },
    "CLDLIQ": {
        "label": "Cloud liquid",
        "units": "kg kg⁻¹",
        "log": True,
        "compare": False,
        "kind": "cloud",
    },
    "CLDICE": {
        "label": "Cloud ice",
        "units": "kg kg⁻¹",
        "log": True,
        "compare": False,
        "kind": "cloud",
    },
}

# Files the analysis script treats as the spun-up Earth O2 grid.
PREFERRED_WACCM = {
    "150": "Earth_150pc_o2.cam.h0.0034-0037.nc",
    "100": "Earth_100pc_o2.cam.h0.0009-0012.nc",
    "50": "Earth_50pc_o2.cam.h0.0040-0043.nc",
    "10": "Earth_10pc_o2.cam.h0.0037-0040.nc",
    "5": "Earth_5pc_o2.cam.h0.0048-0051.nc",
    "1": "Earth_1pc_o2.cam.h0.0045-0048.nc",
    "0.5": "Earth_0.5pc_o2.cam.h0.0055-0058.nc",
    "0.1": "Earth_0.1pc_o2.cam.h0.0033-0036.nc",
}

WACCM_TOKEN = {pal: f"Earth_{pal}pc_o2" for pal in PAL_ORDER}

# Earth transmission files from PSG/Early_Earth_plots.py.
# VULCAN has a 150% PAL Earth spectrum and no 100% PAL Earth spectrum.
EARTH_SPECTRA = {
    "150": {
        "VULCAN": "Earth_150pc_o2_1e12s_48.2SZA_WPT_1rtol_psg_output.txt",
    },
    "100": {
        "Photochem": "Earth_100pc_48.2_psg_output.txt",
        "Atmos": "PTZ_mixingratios_out_psg_output_100pc.txt",
    },
    "10": {
        "VULCAN": "Earth_10pc_o2_1e12s_48.2SZA_WPT_1rtol_psg_output.txt",
        "Photochem": "Earth_10pc_48.2_psg_output.txt",
        "Atmos": "PTZ_mixingratios_out_psg_output_10pc.txt",
    },
    "1": {
        "VULCAN": "Earth_1pc_o2_1e12s_48.2SZA_WPT_1rtol_psg_output.txt",
        "Photochem": "Earth_1pc_48.2_psg_output.txt",
        "Atmos": "PTZ_mixingratios_out_psg_output_1pc.txt",
    },
    "0.1": {
        "VULCAN": "Earth_0.1pc_o2_1e12s_48.2SZA_WPT_1rtol_psg_output.txt",
        "Photochem": "Earth_0.1pc_48.2_psg_output.txt",
        "Atmos": "PTZ_mixingratios_out_psg_output_0.1pc.txt",
    },
}

# Proxima Centauri b transmission files. These are the four PAL cases;
# the paper figure uses the 1% PAL panel.
PCB_SPECTRA = {
    "100": {
        "Photochem": "Proxima_100pc_48.2_psg_output.txt",
        "Atmos": "PTZ_mixingratios_out_psg_output_100pc.txt",
        "WACCM6": "b.e21.BWma1850.f19_g17.PC_b.SSPO.016.cam.h0.0320-0320_psg_output.txt",
        "VULCAN": "PCb_100pc_o2_482SZA_high_g_psg_output.txt",
    },
    "10": {
        "Photochem": "Proxima_10pc_48.2_psg_output.txt",
        "Atmos": "PTZ_mixingratios_out_psg_output_10pc.txt",
        "WACCM6": "b.e21.BWma1850.f19_g17.PC_b.10pc_o2.001.cam.h0.0328_psg_output.txt",
        "VULCAN": "PCb_10pc_o2_482SZA_high_g_psg_output.txt",
    },
    "1": {
        "Photochem": "Proxima_1pc_48.2_psg_output.txt",
        "Atmos": "PTZ_mixingratios_out_psg_output_1pc.txt",
        "WACCM6": "b.e21.BWma1850.f19_g17.PC_b.1pc_o2.002.cam.h0.0338_psg_output.txt",
        "VULCAN": "PCb_1pc_o2_482SZA_high_g_psg_output.txt",
    },
    "0.1": {
        "Photochem": "Proxima_0.1pc_48.2_psg_output.txt",
        "Atmos": "PTZ_mixingratios_out_psg_output_0.1pc.txt",
        "WACCM6": "b.e21.BWma1850.f19_g17.PC_b.SSPO.0.1pc_o2.004.cam.h0.0312_psg_output.txt",
        "VULCAN": "PCb_0.1pc_o2_482SZA_high_g_PT_psg_output.txt",
    },
}

_BARE_EXP = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$")
_K_CGS = 1.380649e-16  # erg K^-1
# Same Boltzmann constant used by n_dens, Atmos_dens, and P_dens in Early_Earth.py.
_K_SI = 1.381e-23  # J K^-1

# Species that are stored as mixing ratios and can also be shown as number density.
CHEMICAL_IDS = ("O3", "O", "O2", "CH4", "N2O", "OH", "H2O", "NOX", "HOX")


def pal_label(pal: str) -> str:
    return f"{pal}% PAL"


def variable_ids(compare_only: bool = False) -> list[str]:
    keys = []
    for key, meta in VARIABLES.items():
        if compare_only and not meta["compare"]:
            continue
        keys.append(key)
    return keys


def _parse_number(token: str) -> float:
    """Parse standard and bare-exponent tokens such as ``1.122-230``."""
    text = token.strip().replace("D", "E").replace("d", "e")
    match = _BARE_EXP.fullmatch(text)
    if match and "e" not in text.lower():
        text = f"{match.group(1)}e{match.group(2)}"
    return float(text)


def _is_header_token(token: str) -> bool:
    try:
        _parse_number(token)
    except ValueError:
        return True
    return False


def read_whitespace_table(path: Path) -> pd.DataFrame:
    """Read a model text table, repairing missing ``e`` in exponents."""
    header = None
    rows = []
    with path.open("r", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("!"):
                continue
            parts = stripped.split()
            if header is None:
                if any(_is_header_token(part) for part in parts):
                    header = parts
                    continue
                header = [f"c{i}" for i in range(len(parts))]
            if len(parts) < len(header):
                continue
            values = []
            valid = True
            for token in parts[: len(header)]:
                try:
                    values.append(_parse_number(token))
                except ValueError:
                    valid = False
                    break
            if valid:
                rows.append(values)
    if header is None or not rows:
        raise ValueError(f"No numeric table in {path}")
    frame = pd.DataFrame(rows, columns=header)
    frame.columns = [str(name).strip() for name in frame.columns]
    return frame


def _column(frame: pd.DataFrame, *names: str) -> np.ndarray | None:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return None


def _sum_species(frame: pd.DataFrame, species: tuple[str, ...], prefixes: tuple[str, ...] = ("", "F")) -> np.ndarray | None:
    total = None
    found = False
    for name in species:
        column = None
        for prefix in prefixes:
            column = _column(frame, f"{prefix}{name}")
            if column is not None:
                break
        if column is None:
            continue
        weight = 2.0 if name == "H2O2" else 1.0
        term = np.nan_to_num(column, nan=0.0) * weight
        total = term if total is None else total + term
        found = True
    if not found:
        return None
    return total


def _sort_profile(pressure: np.ndarray, value: np.ndarray, *extras: np.ndarray | None):
    pressure = np.asarray(pressure, dtype=float)
    value = np.asarray(value, dtype=float)
    mask = np.isfinite(pressure) & (pressure > 0) & np.isfinite(value)
    pressure = pressure[mask]
    value = value[mask]
    cleaned = []
    for extra in extras:
        if extra is None:
            cleaned.append(None)
            continue
        array = np.asarray(extra, dtype=float)
        if array.shape[0] != mask.shape[0]:
            cleaned.append(array)
            continue
        cleaned.append(array[mask])
    order = np.argsort(pressure)
    pressure = pressure[order]
    value = value[order]
    ordered = []
    for array in cleaned:
        if array is None or array.shape[0] != order.shape[0]:
            ordered.append(array)
        else:
            ordered.append(array[order])
    return (pressure, value, *ordered)


def _atmos_sza_dir(pal: str, sza: str) -> Path | None:
    root = REPO_DIR / "Atmos" / f"{pal}pc"
    direct = root / f"SZA_{sza}"
    if (direct / "PTZ_mixingratios_out.dist").is_file():
        return direct
    if sza == "48.2":
        alternate = root / "SZA_48.5"
        if (alternate / "PTZ_mixingratios_out.dist").is_file():
            return alternate
    return None


def _kasting_j_from_outchem(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    lines = path.read_text(errors="replace").splitlines()
    start = None
    for index, line in enumerate(lines):
        if "PHOTOLYSIS RATES" in line:
            start = index + 2
            break
    if start is None:
        return None
    rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Z") and "PO2" in stripped:
            continue
        parts = stripped.split()
        if len(parts) < 8:
            break
        try:
            rows.append(
                {
                    "Z": _parse_number(parts[0]),
                    "PO2": _parse_number(parts[1]),
                    "PO2D": _parse_number(parts[2]),
                    "PO3": _parse_number(parts[6]),
                    "PO3D": _parse_number(parts[7]),
                }
            )
        except ValueError:
            break
    if not rows:
        return None
    return pd.DataFrame(rows)


def _interp_on(x_out: np.ndarray, x_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    order = np.argsort(x_src)
    x_sorted = np.asarray(x_src, dtype=float)[order]
    y_sorted = np.asarray(y_src, dtype=float)[order]
    valid = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    if valid.sum() < 2:
        return np.full(np.shape(x_out), np.nan)
    return np.interp(x_out, x_sorted[valid], y_sorted[valid], left=np.nan, right=np.nan)


def load_atmos(pal: str, sza: str, var_id: str) -> dict | None:
    folder = _atmos_sza_dir(pal, sza)
    if folder is None:
        return None
    frame = read_whitespace_table(folder / "PTZ_mixingratios_out.dist")
    pressure = _column(frame, "PRESS")
    if pressure is None:
        return None
    pressure_hpa = pressure * 1.0e3  # bar -> hPa
    kind = VARIABLES[var_id]["kind"]
    value = _value_from_table(frame, var_id, kind, model="Atmos")
    if kind == "j":
        rates = folder / "out.O2prates"
        if var_id.startswith("jo3") or not rates.is_file():
            return None
        rate_frame = pd.read_csv(
            rates,
            sep=r"\s+",
            header=None,
            names=["Z", "PO2_1", "PO2_2"],
            engine="python",
        )
        altitude = _column(frame, "ALT")
        if altitude is None:
            return None
        branch_a = _interp_on(altitude, rate_frame["Z"].to_numpy(), rate_frame["PO2_1"].to_numpy())
        branch_b = _interp_on(altitude, rate_frame["Z"].to_numpy(), rate_frame["PO2_2"].to_numpy())
        value = _select_j(var_id, branch_a, branch_b, oxidant="o2")
    if value is None:
        return None
    temperature = _column(frame, "TEMP", "temp", "T")
    air = None if temperature is None else _air_from_pressure(pressure_hpa, temperature)
    return _finish_1d("Atmos", pal, sza, pressure_hpa, value, str(folder), air)


def load_photochem(pal: str, sza: str, var_id: str) -> dict | None:
    if VARIABLES[var_id]["kind"] == "j":
        return None
    path = REPO_DIR / "Photochem" / f"{pal}pc" / f"Earth_{pal}pc_{sza}.txt"
    if not path.is_file():
        return None
    frame = read_whitespace_table(path)
    pressure = _column(frame, "press", "PRESS")
    if pressure is None:
        return None
    pressure_hpa = pressure * 1.0e3  # bar -> hPa
    value = _value_from_table(frame, var_id, VARIABLES[var_id]["kind"], model="Photochem")
    if value is None:
        return None
    temperature = _column(frame, "temp", "TEMP", "T")
    air = None if temperature is None else _air_from_pressure(pressure_hpa, temperature)
    return _finish_1d("Photochem", pal, sza, pressure_hpa, value, str(path), air)


def load_kasting(pal: str, sza: str, var_id: str) -> dict | None:
    folder = REPO_DIR / "Kasting_1D_model" / f"{pal}pc" / f"SZA_{sza}"
    path = folder / "OUTPUT_PLOT.dat"
    if not path.is_file():
        return None
    frame = read_whitespace_table(path)
    pressure = _column(frame, "PRESS")
    if pressure is None:
        return None
    pressure_hpa = pressure / 1.0e3  # dyne cm^-2 -> hPa
    kind = VARIABLES[var_id]["kind"]
    if kind == "mixing":
        kasting_names = {
            "O3": "FO3",
            "O": "FO",
            "H2O": "FH2O",
            "CH4": "FCH4",
            "N2O": "FN2O",
            "OH": "FOH",
        }
        if var_id == "O2":
            number = _column(frame, "NumO2")
            density = _column(frame, "DEN")
            if number is None or density is None:
                return None
            value = number / density
        else:
            value = _column(frame, kasting_names.get(var_id, var_id))
    elif kind == "j":
        outchem = _kasting_j_from_outchem(folder / "outchem.dat")
        z_plot = _column(frame, "Z")
        po2 = _column(frame, "PO2")
        po2d = _column(frame, "PO2D")
        po3d = _column(frame, "PO3D")
        po3 = None
        if outchem is not None and z_plot is not None:
            po3 = _interp_on(z_plot, outchem["Z"].to_numpy(), outchem["PO3"].to_numpy())
        if var_id.startswith("jo2"):
            value = _select_j(var_id, po2d, po2, oxidant="o2")
        else:
            value = _select_j(var_id, po3d, po3, oxidant="o3")
    elif kind == "temperature":
        density = _column(frame, "DEN")
        if density is None:
            return None
        value = pressure / (density * _K_CGS)
    else:
        value = _value_from_table(frame, var_id, kind, model="Kasting")
    if value is None:
        return None
    number_density = _column(frame, "DEN")
    air = None if number_density is None else np.asarray(number_density, dtype=float) * 1.0e6
    return _finish_1d("Kasting", pal, sza, pressure_hpa, value, str(path), air)


def _vulcan_dirs() -> list[Path]:
    candidates = []
    env = os.environ.get("VULCAN_OUTPUT")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO_DIR / "VULCAN" / "output")
    candidates.append(Path.home() / "VULCAN" / "output")
    found = []
    for path in candidates:
        if path.is_dir() and path not in found:
            found.append(path)
    return found


def _sza_in_name(name: str, sza: str) -> bool:
    if sza == "48.2":
        return bool(re.search(r"48\.2SZA|482SZA", name))
    return bool(re.search(rf"(?<![\d.]){re.escape(sza)}SZA", name))


def find_vulcan_file(pal: str, sza: str, network: str = "NCHO") -> Path | None:
    patterns = []
    for suffix in (".npz", ".vul"):
        patterns.extend(
            [
                f"Earth_{pal}pcPAL_o2_{sza}SZA*{suffix}",
                f"Earth_{pal}pc_o2_1e12s_{sza}SZA*{suffix}",
                f"Earth_{pal}pc_o2_*{sza}SZA*{suffix}",
            ]
        )
        if pal == "100":
            patterns.append(f"Earth_1e12s_{sza}SZA*{suffix}")
    skip = ("CH4", "hum", "Temp", "Kasting", "high_g", "O3", "C24")
    matches = []
    for folder in _vulcan_dirs():
        for pattern in patterns:
            matches.extend(folder.glob(pattern))
    kept = []
    for path in matches:
        name = path.name
        if not _sza_in_name(name, sza):
            continue
        if any(token in name for token in skip):
            continue
        kept.append(path)
    if not kept:
        return None
    pal_named = [path for path in kept if "PAL" in path.name]
    pool = pal_named or kept
    token = "SNCHOAr_LBC_WPT" if network == "SNCHOAr" else "_NCHO_LBC_WPT"
    network_files = [path for path in pool if token in path.name]
    if not network_files:
        return None
    condensed = [path for path in network_files if path.suffix == ".npz"]
    chosen = condensed or network_files
    return sorted(chosen, key=lambda path: path.name)[0]


def _read_vulcan_chemical(path: Path) -> dict:
    """Final chemistry from a condensed npz or a full .vul pickle."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            species = [str(name) for name in archive["species"]]
            j_table = {
                (str(name), int(branch)): np.asarray(rate, dtype=float)
                for name, branch, rate in zip(
                    archive["j_species"], archive["j_branch"], archive["j_rate"]
                )
            }
            return {
                "species": species,
                "ymix": np.asarray(archive["ymix"], dtype=float),
                "y": np.asarray(archive["y"], dtype=float),
                "pco": np.asarray(archive["pco"], dtype=float),
                "Tco": np.asarray(archive["Tco"], dtype=float),
                "J_sp": j_table,
            }
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    variable = dataset["variable"]
    atmosphere = dataset["atm"]
    return {
        "species": list(variable["species"]),
        "ymix": np.asarray(variable["ymix"], dtype=float),
        "y": np.asarray(variable["y"], dtype=float),
        "pco": np.asarray(atmosphere["pco"], dtype=float),
        "Tco": np.asarray(atmosphere["Tco"], dtype=float),
        "J_sp": variable["J_sp"],
    }


def load_vulcan(pal: str, sza: str, var_id: str, network: str = "NCHO") -> dict | None:
    path = find_vulcan_file(pal, sza, network)
    if path is None:
        return None
    chemical = _read_vulcan_chemical(path)
    species = chemical["species"]
    mixing = chemical["ymix"]
    pressure_hpa = chemical["pco"] / 1.0e3
    kind = VARIABLES[var_id]["kind"]
    if kind == "j":
        j_table = chemical["J_sp"]
        if var_id.startswith("jo2"):
            # Branch 0 is the sum of the later branches.
            branch_b = np.asarray(j_table[("O2", 1)], dtype=float)
            branch_a = np.asarray(j_table[("O2", 2)], dtype=float)
            value = _select_j(var_id, branch_a, branch_b, oxidant="o2")
        else:
            branch_b = np.asarray(j_table[("O3", 1)], dtype=float)
            branch_a = np.asarray(j_table[("O3", 2)], dtype=float)
            value = _select_j(var_id, branch_a, branch_b, oxidant="o3")
    elif kind == "temperature":
        value = chemical["Tco"]
    elif kind == "nox":
        value = _sum_mixing(mixing, species, NOX_SPECIES, h2o2_weight=1.0)
    elif kind == "hox":
        value = _sum_mixing(mixing, species, HOX_SPECIES, h2o2_weight=2.0)
    else:
        value = _mixing(mixing, species, var_id)
    if value is None:
        return None
    air = _vulcan_air_m3(chemical, species)
    if air is None:
        air = _air_from_pressure(pressure_hpa, chemical["Tco"])
    label = "VULCAN SNCHOAr" if network == "SNCHOAr" else "VULCAN NCHO"
    return _finish_1d(label, pal, sza, pressure_hpa, value, str(path), air)


def load_vulcan_ncho(pal: str, sza: str, var_id: str) -> dict | None:
    return load_vulcan(pal, sza, var_id, network="NCHO")


def load_vulcan_sncho(pal: str, sza: str, var_id: str) -> dict | None:
    return load_vulcan(pal, sza, var_id, network="SNCHOAr")


def _mixing(mixing: np.ndarray, species: list[str], name: str) -> np.ndarray | None:
    if name not in species:
        return None
    return mixing[:, species.index(name)]


def _sum_mixing(mixing: np.ndarray, species: list[str], names: tuple[str, ...], h2o2_weight: float) -> np.ndarray | None:
    total = None
    for name in names:
        column = _mixing(mixing, species, name)
        if column is None:
            continue
        weight = h2o2_weight if name == "H2O2" else 1.0
        term = np.nan_to_num(column, nan=0.0) * weight
        total = term if total is None else total + term
    return total


def _select_j(var_id: str, branch_a: np.ndarray | None, branch_b: np.ndarray | None, oxidant: str) -> np.ndarray | None:
    """Return one photolysis frequency.

    ``branch_a`` is the O(1D) channel for O3 (WACCM ``jo3_a``, Kasting ``PO3D``,
    VULCAN branch 2) and the second O2 column for O2. ``branch_b`` is the
    complementary channel. Totals are the sum, without the factor of two used
    when counting odd-oxygen atoms.
    """
    del oxidant
    if var_id.endswith("_a"):
        return None if branch_a is None else np.asarray(branch_a, dtype=float)
    if var_id.endswith("_b"):
        return None if branch_b is None else np.asarray(branch_b, dtype=float)
    if branch_a is None and branch_b is None:
        return None
    if branch_a is None:
        return np.asarray(branch_b, dtype=float)
    if branch_b is None:
        return np.asarray(branch_a, dtype=float)
    return np.asarray(branch_a, dtype=float) + np.asarray(branch_b, dtype=float)


def _value_from_table(frame: pd.DataFrame, var_id: str, kind: str, model: str) -> np.ndarray | None:
    del model
    if kind == "nox":
        return _sum_species(frame, NOX_SPECIES)
    if kind == "hox":
        return _sum_species(frame, HOX_SPECIES)
    if kind == "temperature":
        return _column(frame, "TEMP", "temp", "T")
    if kind == "mixing":
        return _column(frame, var_id, var_id.lower())
    return None


def _air_from_pressure(pressure_hpa, temperature) -> np.ndarray:
    """Ideal-gas number density, molecules m^-3. Pressure is in hPa."""
    pressure = np.asarray(pressure_hpa, dtype=float).reshape(-1)
    temp = np.asarray(temperature, dtype=float).reshape(-1)
    count = min(pressure.shape[0], temp.shape[0])
    air = np.full(count, np.nan)
    p = pressure[:count]
    t = temp[:count]
    ok = np.isfinite(p) & np.isfinite(t) & (t > 0) & (p > 0)
    air[ok] = p[ok] * 100.0 / (_K_SI * t[ok])
    return air


def _vulcan_air_m3(variable: dict, species: list[str]) -> np.ndarray | None:
    """VULCAN stores number density in cm^-3. Convert with an abundant species."""
    if "y" not in variable or "ymix" not in variable:
        return None
    number = np.asarray(variable["y"], dtype=float)
    mixing = np.asarray(variable["ymix"], dtype=float)
    for name in ("N2", "O2", "CO2", "H2O"):
        if name not in species:
            continue
        index = species.index(name)
        mix = mixing[:, index]
        num = number[:, index]
        air = np.full(mix.shape, np.nan)
        ok = np.isfinite(mix) & np.isfinite(num) & (np.abs(mix) > 0)
        air[ok] = num[ok] / mix[ok] * 1.0e6
        if np.isfinite(air).sum() > 2:
            return air
    return None


def _finish_1d(model: str, pal: str, sza: str, pressure_hpa, value, source: str, air_m3=None) -> dict | None:
    pressure = np.asarray(pressure_hpa, dtype=float)
    values = np.asarray(value, dtype=float)
    air = None if air_m3 is None else np.asarray(air_m3, dtype=float)
    if pressure.shape[0] != values.shape[0]:
        count = min(pressure.shape[0], values.shape[0])
        pressure = pressure[:count]
        values = values[:count]
        if air is not None:
            air = air[:count]
    if air is not None and air.shape[0] != pressure.shape[0]:
        air = None
    if air is None:
        pressure, values = _sort_profile(pressure, values)[:2]
    else:
        pressure, values, air = _sort_profile(pressure, values, air)
    if pressure.size == 0:
        return None
    profile = {
        "model": model,
        "pal": pal,
        "sza": sza,
        "pressure_hpa": pressure,
        "value": values,
        "lat": None,
        "zonal": None,
        "zonal_pressure_hpa": None,
        "source": source,
        "air_m3": air,
        "air_zonal": None,
        "density": None,
        "density_zonal": None,
    }
    return profile


def _waccm_dirs() -> list[Path]:
    """Compressed WACCM6 subsets shipped with the repository."""
    candidates = []
    env = os.environ.get("WACCM_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(WACCM_SUBSET_DIR)
    found = []
    for path in candidates:
        if path.is_dir() and path not in found:
            found.append(path)
    return found


def find_waccm_file(pal: str) -> Path | None:
    token = WACCM_TOKEN[pal]
    preferred_names = {
        PREFERRED_WACCM[pal],
        PREFERRED_WACCM[pal].replace(".nc", "_subset.nc"),
    }
    skip = ("_dyn", "_YS", "NPZD", "scat", "Monthly")
    for folder in _waccm_dirs():
        hits = []
        for path in folder.glob("*.nc"):
            name = path.name
            if any(token_skip in name for token_skip in skip):
                continue
            if token not in name:
                continue
            hits.append(path)
        if not hits:
            continue
        for path in hits:
            if path.name in preferred_names:
                return path
        subsets = [path for path in hits if "subset" in path.name]
        pool = subsets or [path for path in hits if re.search(r"h0\.\d+-\d+", path.name)] or hits
        return sorted(pool, key=lambda item: item.name)[-1]
    return None


def _waccm_names(var_id: str) -> list[str]:
    if var_id in ("jo2", "jo3"):
        return [f"{var_id}_a", f"{var_id}_b"]
    return [var_id]


def load_waccm(pal: str, sza: str, var_id: str) -> dict | None:
    """Global-mean and zonal-mean profile for one WACCM6 history file.

    ``sza`` is unused: WACCM6 is a 3D climate run, not a fixed-zenith 1D case.
    """
    del sza
    path = find_waccm_file(pal)
    if path is None:
        return None
    import xarray as xr

    names = _waccm_names(var_id)
    with xr.open_dataset(path, decode_times=False) as dataset:
        missing = [name for name in names if name not in dataset]
        if missing:
            return None
        fields = []
        for name in names:
            data = _zonal_field(dataset[name])
            fields.append(data)
        data = fields[0] if len(fields) == 1 else fields[0] + fields[1]
        gw = dataset["gw"]
        if "lat" in data.dims:
            zonal = data
            global_mean = (data * gw).sum("lat") / gw.sum("lat")
            latitude = np.asarray(dataset["lat"].values, dtype=float)
        else:
            zonal = None
            global_mean = data
            latitude = None
        pressure_hpa, zonal_pressure = _waccm_pressure(dataset)
        value = np.asarray(global_mean.values, dtype=float).reshape(-1)
        zonal_values = None if zonal is None else np.asarray(zonal.values, dtype=float)
        density = None
        density_zonal = None
        if "T" in dataset and zonal_values is not None:
            temperature = _zonal_field(dataset["T"])
            t_zonal = np.asarray(temperature.values, dtype=float)
            if t_zonal.shape == zonal_values.shape:
                if zonal_pressure is not None and np.asarray(zonal_pressure).shape == zonal_values.shape:
                    p_zonal = np.asarray(zonal_pressure, dtype=float)
                else:
                    p_zonal = pressure_hpa[:, None]
                density_zonal = zonal_values * (p_zonal * 100.0) / (_K_SI * t_zonal)
                weights = np.asarray(gw.values, dtype=float)
                density = np.nansum(density_zonal * weights, axis=1) / np.nansum(weights)
    if zonal_values is not None and zonal_values.shape[0] == pressure_hpa.shape[0]:
        order = np.argsort(pressure_hpa)
        pressure_hpa = pressure_hpa[order]
        value = value[order]
        zonal_values = zonal_values[order, :]
        if density is not None:
            density = density[order]
        if density_zonal is not None:
            density_zonal = density_zonal[order, :]
        if zonal_pressure is not None and zonal_pressure.ndim == 2:
            zonal_pressure = zonal_pressure[order, :]
        elif zonal_pressure is not None:
            zonal_pressure = zonal_pressure[order]
    else:
        pressure_hpa, value = _sort_profile(pressure_hpa, value)[:2]
        zonal_values = None
        zonal_pressure = None
        density = None
        density_zonal = None
    if pressure_hpa.size == 0:
        return None
    return {
        "model": "WACCM6",
        "pal": pal,
        "sza": "3D",
        "pressure_hpa": pressure_hpa,
        "value": value,
        "lat": latitude,
        "zonal": zonal_values,
        "zonal_pressure_hpa": zonal_pressure,
        "source": str(path),
        "air_m3": None,
        "air_zonal": None,
        "density": density,
        "density_zonal": density_zonal,
    }


def _zonal_field(data):
    """Collapse time and longitude. A size-1 ``zlon`` left by ``ncwa`` is dropped."""
    if "time" in data.dims:
        data = data.mean("time")
    for name in ("lon", "zlon"):
        if name not in data.dims:
            continue
        if data.sizes[name] == 1:
            data = data.squeeze(name, drop=True)
        else:
            data = data.mean(name)
    return data


def _waccm_pressure(dataset) -> tuple[np.ndarray, np.ndarray | None]:
    needed = ("hyam", "hybm", "P0", "PS", "gw")
    if not all(name in dataset for name in needed):
        pressure = np.asarray(dataset["lev"].values, dtype=float)
        return pressure, None
    surface = _zonal_field(dataset["PS"])
    gw = dataset["gw"]
    surface_mean = (surface * gw).sum("lat") / gw.sum("lat")
    pressure_mean = (dataset["hyam"] * dataset["P0"] + dataset["hybm"] * surface_mean) / 100.0
    pressure_zonal = (dataset["hyam"] * dataset["P0"] + dataset["hybm"] * surface) / 100.0
    return (
        np.asarray(pressure_mean.values, dtype=float).reshape(-1),
        np.asarray(pressure_zonal.values, dtype=float),
    )


_LOADERS = {
    "WACCM6": load_waccm,
    "Atmos": load_atmos,
    "Photochem": load_photochem,
    "VULCAN": load_vulcan_ncho,
    "VULCAN NCHO": load_vulcan_ncho,
    "VULCAN SNCHOAr": load_vulcan_sncho,
    "Kasting": load_kasting,
}


def as_number_density(profile: dict | None) -> dict | None:
    """Mixing-ratio profile converted to molecules m^-3."""
    if profile is None:
        return None
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


def load_o2_photolysis_rate(model: str, pal: str, sza: str) -> dict | None:
    """Odd-oxygen production from O2 photolysis, 2 J(O2) n(O2).

    This is the rate plotted in Early_Earth.py (`prox_ox_W`, `prox_ox_K`,
    `prox_ox_V`, and Atmos `2 (PO2_1 + PO2_2) n`). Units are molecules m^-3 s^-1.
    """
    j_profile = _LOADERS[model](pal, sza, "jo2")
    o2_density = as_number_density(_LOADERS[model](pal, sza, "O2"))
    if j_profile is None or o2_density is None:
        return None
    count = min(np.asarray(j_profile["value"]).shape[0], np.asarray(o2_density["value"]).shape[0])
    value = 2.0 * np.asarray(j_profile["value"], dtype=float)[:count] * np.asarray(o2_density["value"], dtype=float)[:count]
    zonal = None
    j_zonal = j_profile.get("zonal")
    n_zonal = o2_density.get("zonal")
    if j_zonal is not None and n_zonal is not None:
        rows = min(count, np.asarray(j_zonal).shape[0], np.asarray(n_zonal).shape[0])
        cols = min(np.asarray(j_zonal).shape[1], np.asarray(n_zonal).shape[1])
        zonal = 2.0 * np.asarray(j_zonal, dtype=float)[:rows, :cols] * np.asarray(n_zonal, dtype=float)[:rows, :cols]
        count = rows
        value = value[:count]
    result = dict(j_profile)
    result["pressure_hpa"] = np.asarray(j_profile["pressure_hpa"], dtype=float)[:count]
    result["value"] = value
    result["zonal"] = zonal
    result["density"] = None
    result["density_zonal"] = None
    result["air_m3"] = None
    result["air_zonal"] = None
    return result


def load_profile(model: str, pal: str, sza: str, var_id: str) -> dict | None:
    if var_id not in VARIABLES:
        raise KeyError(var_id)
    if model not in _LOADERS:
        raise KeyError(model)
    if not VARIABLES[var_id]["compare"] and model != "WACCM6":
        return None
    if var_id == "jo2_rate":
        return load_o2_photolysis_rate(model, pal, sza)
    return _LOADERS[model](pal, sza, var_id)


def source_stamp(model: str, pal: str, sza: str, var_id: str) -> str:
    """Cache key that changes when the underlying file changes."""
    if var_id == "jo2_rate":
        return source_stamp(model, pal, sza, "jo2") + "|" + source_stamp(model, pal, sza, "O2")
    path = None
    if model == "WACCM6":
        path = find_waccm_file(pal)
    elif model == "Atmos":
        folder = _atmos_sza_dir(pal, sza)
        path = None if folder is None else folder / "PTZ_mixingratios_out.dist"
        if folder is not None and var_id.startswith("jo2"):
            rates = folder / "out.O2prates"
            if path is not None and path.is_file() and rates.is_file():
                rate_stat = rates.stat()
                file_stat = path.stat()
                return (
                    f"{path}:{file_stat.st_mtime_ns}:{file_stat.st_size}"
                    f"|{rates}:{rate_stat.st_mtime_ns}:{rate_stat.st_size}"
                )
    elif model == "Photochem":
        path = REPO_DIR / "Photochem" / f"{pal}pc" / f"Earth_{pal}pc_{sza}.txt"
    elif model in ("VULCAN", "VULCAN NCHO", "VULCAN SNCHOAr"):
        network = "SNCHOAr" if model == "VULCAN SNCHOAr" else "NCHO"
        path = find_vulcan_file(pal, sza, network)
    elif model == "Kasting":
        path = REPO_DIR / "Kasting_1D_model" / f"{pal}pc" / f"SZA_{sza}" / "OUTPUT_PLOT.dat"
    if path is None or not path.is_file():
        return f"missing:{model}:{pal}:{sza}:{var_id}"
    stat = path.stat()
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


# Oxygen levels on the paper O2–O3 curve, low to high. x = 1 is 100% PAL.
CURVE_PALS = ["0.1", "0.5", "1", "5", "10", "50", "100", "150"]
_DU_PER_M2 = 2.6867e20
_GRAVITY = 9.81
_AMU = 1.661e-27


def oxygen_mixing_ratio(pal: str) -> float:
    """O2 volume mixing ratio. 100% PAL is 0.21."""
    return 0.21 * float(pal) / 100.0


def _mean_molecular_mass(o2_mr: float) -> float:
    """Dry-air mass per molecule, matching Early_Earth.py ``O3_col``."""
    return ((28.0 * (1.0 - o2_mr)) + (o2_mr * 32.0)) * _AMU


def ozone_column_du(pressure_hpa, mixing, o2_mr: float) -> float:
    """Ozone column in Dobson units from mixing ratio and pressure.

    This is the pressure integral used for the WACCM6 columns:
    N = ∫ χ dp / (m g), then divide by 2.6867×10²⁰ m⁻².
    """
    pressure = np.asarray(pressure_hpa, dtype=float).reshape(-1) * 100.0
    chi = np.asarray(mixing, dtype=float).reshape(-1)
    count = min(pressure.size, chi.size)
    pressure = pressure[:count]
    chi = chi[:count]
    ok = np.isfinite(pressure) & np.isfinite(chi) & (pressure > 0)
    if ok.sum() < 2:
        return float("nan")
    order = np.argsort(pressure[ok])
    integral = float(np.trapz(chi[ok][order], pressure[ok][order]))
    column = abs(integral) / (_mean_molecular_mass(o2_mr) * _GRAVITY)
    return column / _DU_PER_M2


def waccm_ozone_column(pal: str) -> tuple[float, float, float]:
    """Gaussian-mean ozone column and the min/max over latitude and longitude.

    Interface pressures follow ``O3_col`` in early_earth_lib.py. A zonal-mean
    file has no longitude, so the range is across latitude only.
    """
    path = find_waccm_file(pal)
    if path is None:
        return float("nan"), float("nan"), float("nan")
    import xarray as xr

    o2_mr = oxygen_mixing_ratio(pal)
    mass = _mean_molecular_mass(o2_mr)
    with xr.open_dataset(path, decode_times=False) as dataset:
        if not all(name in dataset for name in ("O3", "PS", "hyai", "hybi", "P0", "gw")):
            return float("nan"), float("nan"), float("nan")
        ozone = dataset["O3"]
        surface = dataset["PS"]
        if "time" in ozone.dims:
            ozone = ozone.mean("time")
        if "time" in surface.dims:
            surface = surface.mean("time")
        ozone = ozone.squeeze(drop=True)
        surface = surface.squeeze(drop=True)
        hyai = np.asarray(dataset["hyai"].values, dtype=float)
        hybi = np.asarray(dataset["hybi"].values, dtype=float)
        p0 = float(np.asarray(dataset["P0"].values))
        weights = np.asarray(dataset["gw"].values, dtype=float)
        ozone_values = np.asarray(ozone.values, dtype=float)
        surface_values = np.asarray(surface.values, dtype=float)
    surface_values = surface_values.reshape((1,) + surface_values.shape)
    pad = (1,) * (surface_values.ndim - 1)
    interface = hyai.reshape((-1,) + pad) * p0 + hybi.reshape((-1,) + pad) * surface_values
    thickness = interface[1:] - interface[:-1]
    if thickness.shape != ozone_values.shape:
        return float("nan"), float("nan"), float("nan")
    column = np.sum(ozone_values * thickness / mass / _GRAVITY, axis=0) / _DU_PER_M2
    column = np.asarray(column, dtype=float)
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    if column.ndim == 2:
        zonal = np.nanmean(column, axis=-1)
    else:
        zonal = column
    if zonal.shape[0] != weights.shape[0]:
        mean = float(np.nanmean(finite))
    else:
        mean = float(np.nansum(zonal * weights) / np.nansum(weights))
    return mean, float(np.nanmin(finite)), float(np.nanmax(finite))


def model_ozone_column(model: str, pal: str, sza: str) -> float:
    """1D ozone column at one oxygen level and solar zenith angle."""
    profile = load_profile(model, pal, sza, "O3")
    if profile is None:
        return float("nan")
    return ozone_column_du(profile["pressure_hpa"], profile["value"], oxygen_mixing_ratio(pal))


def _column_band(model: str, pal: str, angles: tuple[str, str]) -> tuple[float, float]:
    values = [model_ozone_column(model, pal, angle) for angle in angles]
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    return float(min(finite)), float(max(finite))


def ozone_oxygen_curve() -> dict:
    """Paper O2–O3 curve: WACCM6 and the 1D solar-zenith-angle ranges.

    NCHO uses 48.2° and 60°. SNCHOAr uses 45° and 60°. Those are the angles
    present at every oxygen level. The other 1D models use 45° and 60°.
    """
    bands = {
        "Kasting": ("45", "60"),
        "Photochem": ("45", "60"),
        "Atmos": ("45", "60"),
        "VULCAN NCHO": ("48.2", "60"),
        "VULCAN SNCHOAr": ("45", "60"),
    }
    x = []
    waccm_mean = []
    waccm_min = []
    waccm_max = []
    stored = {name: {"low": [], "high": []} for name in bands}
    envelope_low = []
    envelope_high = []
    notes = []
    for pal in CURVE_PALS:
        x.append(float(pal) / 100.0)
        mean, low, high = waccm_ozone_column(pal)
        waccm_mean.append(mean)
        waccm_min.append(low)
        waccm_max.append(high)
        if not np.isfinite(mean):
            notes.append(f"WACCM6 has no {pal_label(pal)} ozone column.")
        edges = []
        for name, angles in bands.items():
            band_low, band_high = _column_band(name, pal, angles)
            stored[name]["low"].append(band_low)
            stored[name]["high"].append(band_high)
            if np.isfinite(band_low):
                edges.extend([band_low, band_high])
            else:
                notes.append(
                    f"{name} has no {pal_label(pal)} ozone column at "
                    f"{angles[0]}° and {angles[1]}°."
                )
        if edges:
            envelope_low.append(float(min(edges)))
            envelope_high.append(float(max(edges)))
        else:
            envelope_low.append(float("nan"))
            envelope_high.append(float("nan"))
    return {
        "x": x,
        "waccm_mean": waccm_mean,
        "waccm_min": waccm_min,
        "waccm_max": waccm_max,
        "bands": stored,
        "envelope_low": envelope_low,
        "envelope_high": envelope_high,
        "notes": notes,
    }


def load_spectrum_file(filename: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = SPECTRA_DIR / filename
    if not path.is_file():
        alternate = Path.home() / "psg_outputs" / filename
        path = alternate if alternate.is_file() else path
    if not path.is_file():
        return None
    data = np.genfromtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 6:
        return None
    wavelength = np.asarray(data[:, 0], dtype=float)
    altitude = np.asarray(data[:, 5], dtype=float)
    mask = np.isfinite(wavelength) & np.isfinite(altitude)
    return wavelength[mask], altitude[mask]


def spectra_for(planet: str, pal: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    table = PCB_SPECTRA if planet == "proxima" else EARTH_SPECTRA
    found = {}
    for model, filename in table.get(pal, {}).items():
        loaded = load_spectrum_file(filename)
        if loaded is not None:
            found[model] = loaded
    return found


def slice_zonal(profile: dict, pressure_hpa: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Zonal-mean value against latitude at one pressure."""
    zonal = profile.get("zonal")
    latitude = profile.get("lat")
    if zonal is None or latitude is None:
        return None
    pressure = profile.get("zonal_pressure_hpa")
    if pressure is None:
        pressure = profile["pressure_hpa"]
    pressure = np.asarray(pressure, dtype=float)
    zonal = np.asarray(zonal, dtype=float)
    if pressure.ndim == 2:
        pressure = np.nanmean(pressure, axis=1)
    order = np.argsort(pressure)
    pressure = pressure[order]
    zonal = zonal[order, :]
    valid = np.isfinite(pressure) & (pressure > 0)
    pressure = pressure[valid]
    zonal = zonal[valid, :]
    if pressure.size < 2:
        return None
    if pressure_hpa < float(np.nanmin(pressure)) or pressure_hpa > float(np.nanmax(pressure)):
        return None
    log_p = np.log(pressure)
    target = np.log(pressure_hpa)
    sliced = np.array([np.interp(target, log_p, zonal[:, index]) for index in range(zonal.shape[1])])
    return np.asarray(latitude, dtype=float), sliced
