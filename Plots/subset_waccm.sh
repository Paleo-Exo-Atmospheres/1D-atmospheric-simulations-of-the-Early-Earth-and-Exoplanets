#!/bin/bash
# Keep the Early Earth WACCM6 fields used by Plots/streamlit_app.py.
#
# GitHub rejects files larger than 100 MB. These history files are ~500 MB
# with the full variable list. ncks drops everything else and compresses
# the remainder. Coordinate variables (time, lev, lat, lon, ilev) are kept
# automatically. PS, hyam, hybm, P0, and gw are required to build pressure
# and the latitude-weighted global mean.
#
# Usage:
#   ./subset_waccm.sh
#   ./subset_waccm.sh /path/to/CESM_data
#
# Output lands in Plots/waccm/, which is where the app looks first.

set -euo pipefail

SRC="${1:-${HOME}/CESM_data}"
DST="$(cd "$(dirname "$0")" && pwd)/waccm"
mkdir -p "${DST}"

VARS="U,V,T,O2,O3,O,CH4,N2O,NOX,HOX,OH,H2O,CLDLIQ,CLDICE,jo2_a,jo2_b,jo3_a,jo3_b,PS,hyam,hybm,P0,gw"

files=(
  Earth_100pc_o2.cam.h0.0009-0012.nc
  Earth_50pc_o2.cam.h0.0040-0043.nc
  Earth_10pc_o2.cam.h0.0037-0040.nc
  Earth_5pc_o2.cam.h0.0048-0051.nc
  Earth_1pc_o2.cam.h0.0045-0048.nc
  Earth_0.5pc_o2.cam.h0.0055-0058.nc
  Earth_0.1pc_o2.cam.h0.0033-0036.nc
)

for name in "${files[@]}"; do
  in="${SRC}/${name}"
  out="${DST}/${name%.nc}_subset.nc"
  if [[ ! -f "${in}" ]]; then
    echo "missing ${in}" >&2
    continue
  fi
  echo "subset ${name}"
  ncks -O -4 -L 4 -v "${VARS}" "${in}" "${out}"
done
