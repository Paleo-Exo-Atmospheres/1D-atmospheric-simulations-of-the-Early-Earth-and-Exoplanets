"""Write the final VULCAN chemistry from Earth LBC + WPT runs.

The .vul pickles are about 18 MB because they store wavelength-dependent
optical depths, fluxes, and cross sections. This script keeps the converged
chemical state: every species mixing ratio and number density, temperature,
pressure, the final integration time, and every saved photolysis frequency.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

SOURCE_DIR = Path("/Users/gregcooke/VULCAN/output")
DEST_DIR = Path(__file__).resolve().parent / "output"


def condense_vulcan(path: Path, dest: Path) -> None:
    """Save the final chemical state of one .vul file."""
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    variable = dataset["variable"]
    atmosphere = dataset["atm"]
    species = np.array([str(name) for name in variable["species"]])
    photolysis = variable["J_sp"]
    names = []
    branches = []
    rates = []
    for key, rate in photolysis.items():
        names.append(str(key[0]))
        branches.append(int(key[1]))
        rates.append(np.asarray(rate, dtype=np.float64))
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        species=species,
        ymix=np.asarray(variable["ymix"], dtype=np.float64),
        y=np.asarray(variable["y"], dtype=np.float64),
        pco=np.asarray(atmosphere["pco"], dtype=np.float64),
        Tco=np.asarray(atmosphere["Tco"], dtype=np.float64),
        t=np.asarray(variable["t"], dtype=np.float64),
        j_species=np.array(names),
        j_branch=np.asarray(branches, dtype=np.int32),
        j_rate=np.vstack(rates),
        source=np.array(path.name),
    )


def earth_lbc_wpt_files(source_dir: Path) -> list[Path]:
    """Earth PAL runs that use a lower boundary condition and the WACCM temperature profile."""
    files = []
    for path in sorted(source_dir.glob("Earth_*PAL*_LBC_WPT_*.vul")):
        files.append(path)
    return files


def main() -> None:
    files = earth_lbc_wpt_files(SOURCE_DIR)
    if not files:
        raise SystemExit(f"No Earth LBC WPT files in {SOURCE_DIR}")
    for path in files:
        dest = DEST_DIR / (path.stem + ".npz")
        condense_vulcan(path, dest)
        print(f"{dest.name}  {dest.stat().st_size / 1024:.0f} KB  from {path.name}")
    print(f"Wrote {len(files)} files to {DEST_DIR}")


if __name__ == "__main__":
    main()
