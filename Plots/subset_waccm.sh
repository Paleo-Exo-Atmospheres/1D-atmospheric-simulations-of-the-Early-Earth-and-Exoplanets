#!/bin/bash
# Zonal-mean Early Earth WACCM6 fields for Plots/streamlit_app.py.
#
# ncks keeps the chemistry, winds, clouds, and the hybrid-pressure
# coefficients, then ncwa averages longitude. The full-longitude subset is
# only a temporary file. Output is *_subset_zonal.nc in the matching
# WACCM6 oxygen folder.
#
# 10% to 0.1% PAL use the corrected upper-boundary (ubc) history files.
#
# Usage:
#   ./subset_waccm.sh
#   ./subset_waccm.sh /path/to/CESM_data

set -euo pipefail

SRC="${1:-${HOME}/CESM_data}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WACCM6="${ROOT}/WACCM6"

VARS="U,V,T,O2,O3,O,CH4,N2O,NOX,HOX,OH,H2O,CLDLIQ,CLDICE,jo2_a,jo2_b,jo3_a,jo3_b,PS,hyam,hybm,hyai,hybi,P0,gw"

# name|oxygen folder under WACCM6
files=(
  "Earth_150pc_o2.cam.h0.0034-0037.nc|150pc"
  "Earth_100pc_o2.cam.h0.0009-0012.nc|100pc"
  "Earth_50pc_o2.cam.h0.0040-0043.nc|50pc"
  "Earth_10pc_o2_ubc.cam.h0.0036.nc|10pc"
  "Earth_5pc_o2_ubc.cam.h0.0047.nc|5pc"
  "Earth_1pc_o2_ubc.cam.h0.0044.nc|1pc"
  "Earth_0.5pc_o2_ubc.cam.h0.0056.nc|0.5pc"
  "Earth_0.1pc_o2_ubc.cam.h0.0045.nc|0.1pc"
)

for item in "${files[@]}"; do
  name="${item%%|*}"
  folder="${item##*|}"
  src_file="${SRC}/${name}"
  if [[ ! -f "${src_file}" ]]; then
    echo "missing ${src_file}" >&2
    continue
  fi
  mkdir -p "${WACCM6}/${folder}"
  zonal="${WACCM6}/${folder}/${name%.nc}_subset_zonal.nc"
  tmp="${WACCM6}/${folder}/.${name%.nc}_subset.nc"
  kept=""
  header="$(ncdump -h "${src_file}")"
  for var in ${VARS//,/ }; do
    if printf '%s\n' "${header}" | grep -Eq " ${var}[(;]"; then
      kept="${kept:+${kept},}${var}"
    fi
  done
  echo "zonal ${name}"
  echo "  variables ${kept}"
  ncks -O -4 -L 4 -v "${kept}" "${src_file}" "${tmp}"
  ncwa -O -a lon -4 -L 4 "${tmp}" "${zonal}"
  rm -f "${tmp}"
done
