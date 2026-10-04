# 1D Simulations of the Early Earth and Exoplanets

**Authors:** Akhil Kumar & Gregory J. Cooke  
**Affiliation:** Institute of Astronomy, University of Cambridge

## Overview

This repository accompanies the paper **"Simulations of the Evolving Ozone Layer: Implications for the Early Earth, Exoplanets, and the Search for Life"** It contains the simulation data, model configuration files, and analysis scripts used to study how Earth's atmosphere — and in particular its ozone (O₃) layer — responds to changing atmospheric oxygen levels from the **Proterozoic through the Phanerozoic to the present day**.

How does atmospheric photochemistry and ozone evolve as O₂ is increased from 0.1% the present atmospheric level (PAL) to modern day concentrations? This is a question that has been investigated for over 5 decades. Models have been run independently, but we wanted to see how models compared when starting from the same boundary conditions.

In terms of Earth's history, the rise in oxygen we are considering roughly corresponds to the early Proterozoic (~2.4 Gyr onward) through the oxygenation events of the Phanerozoic to modern conditions. We simulate this evolution using four independent one-dimensional (1D) photochemical models, compared against **WACCM6**, a 3D chemistry-climate model. WACCM6 is a high-top extension of a configuration of the Community Earth System Model (CESM2) that includes interactive atmospheric chemistry.

We systematically vary:

1. **Atmospheric oxygenation** — O₂ boundary conditions from 0.1% to 150% of Present Atmospheric Level (PAL), spanning Proterozoic anoxia through Phanerozoic and modern oxygenation.
2. **Solar evolution** — the changing solar UV spectrum and bolometric flux from 4.0 Ga to the present day (the faint young Sun problem) in some of the 1D models.
3. **Thermal structure** — pressure–temperature (P–T) profiles taken from WACCM6 for each oxygenation state, ensuring the 1D models use atmospheres somewhat consistent with the 3D climate.
4. **Lower-boundary fluxes** — surface emissions of CH₄, N₂O, and related reduced gases.
5. **Water vapour** — tropospheric humidity scaling (e.g. 40% relative humidity perturbations).
6. **Surface albedo** — values from 0.06 to 0.30.
7. **Solar zenith angle (SZA)** — 45°, 48.2°, and 60°, with appropriate diurnal averaging factors.

We quantify how model choice and boundary-condition assumptions affect ozone columns, odd-oxygen budgets, and synthetic transmission/emission spectra relevant to Earth-like exoplanet observations.

---

## Models used

| Model | Description | Repository |
|-------|-------------|------------|
| **Atmos** | 1D photochemical–climate model from the Virtual Planetary Laboratory | [github.com/VirtualPlanetaryLaboratory/atmos](https://github.com/VirtualPlanetaryLaboratory/atmos) |
| **Photochem** | Photochemical and climate model for planetary atmospheres | [github.com/Nicholaswogan/photochem](https://github.com/Nicholaswogan/photochem) |
| **VULCAN** | Chemical kinetics code for exoplanetary and planetary atmospheres | [github.com/shami-EEG/VULCAN](https://github.com/shami-EEG/VULCAN) |
| **Kasting 1D model** | 1D photochemical model developed in the group headed by James Kasting, with correlated-*k* O₂ Schumann–Runge photolysis (Ji et al. 2024) | [zenodo.org/records/10822978](https://zenodo.org/records/10822978) |
| **WACCM6** | Whole Atmosphere Community Climate Model version 6 — a configuration of CESM used as the 3D benchmark | [github.com/ESCOMP/CESM](https://github.com/ESCOMP/CESM) |

All four 1D models are run with harmonised boundary conditions where possible (matching lower-boundary mixing ratios, stellar spectra, and WACCM6 P–T profiles) so that differences in output reflect genuine model physics rather than inconsistent inputs.

---

## Oxygenation states

Simulations are organised by O₂ level expressed as a percentage of PAL:

| Folder name | Approx. O₂ (% PAL) |
|-------------|-------------------|
| `0.1pc` | 0.1% |
| `0.5pc` | 0.5% |
| `1pc` | 1% |
| `5pc` | 5% |
| `10pc` | 10% |
| `50pc` | 50% |
| `100pc` | 100% |
| `150pc` | 150% |

Solar-age labels such as `Sun_0.0Ga` (present Sun) and `Sun_2.4Ga` (younger Sun) indicate which stellar spectrum was used. The **Young Sun** and **Coupled Young Sun Simulations** folders explore the faint young Sun explicitly.

---

## Repository structure

```
1D-Simulations-of-the-Early-Earth/
├── Atmos/                  # Atmos model output
├── Photochem/              # Photochem model output
├── VULCAN/                 # VULCAN configuration and input files
├── Kasting_1D_model/       # Kasting 1D model output
├── WACCM6/                 # WACCM6 files
├── Python/                 # Analysis and plotting scripts
├── PSG/                    # Planetary Spectrum Generator post-processing
├── Plots/                  # Interactive explorer
└── README.md
```

### `Atmos/`

Output from the [Atmos](https://github.com/VirtualPlanetaryLaboratory/atmos) model. Simulations are **uncoupled** (fixed P–T profiles from WACCM6).

**Standard grid** (`0.1pc/`, `1pc/`, … `150pc/`):  
Each oxygenation folder contains sub-folders for solar zenith angle (`SZA_45`, `SZA_48.2` or `SZA_48.5`, `SZA_60`). Diurnal averaging factors are 0.354 (45°), 0.375 (48.2°), and 0.5 (60°). Typical output files per run:

- `PTZ_mixingratios_out.dist` — altitude, pressure, temperature, and species mixing ratios
- `out.O2prates` — O₂ photolysis rates
- `int.rates.out` — integrated photochemical rates

**`WACCM_PT_profiles/`** — P–T profiles extracted from WACCM6, used as input to Atmos.

**`Methane_Perturbations/`** — CH₄ lower-boundary flux varied at selected O₂ levels (`0.1pc_PT_profile`, `1pc_PT_profile`, `10pc_PT_profile`, `100pc_PT_profile`).

**`Albedo Perturbations/`** — surface albedo varied (0.06–0.30) at fixed SZA.

**`Coupled Young Sun Simulations/`** — young-Sun stellar spectra at 0.1% and 100% PAL for ages 0.0 Ga and 2.4 Ga.

**`Proxima Centauri/`** — Earth-like boundary conditions with Proxima Centauri b P–T profiles and stellar flux (GJ 551).

**`Old sims/`** — early test runs with an outdated O₃ deposition velocity; retained for reference but **not recommended for analysis**.

---

### `Photochem/`

Output from [Photochem](https://github.com/Nicholaswogan/photochem). The parameter space mirrors Atmos: WACCM6 P–T profiles, solar evolution, and boundary-condition perturbations.

**Standard grid** (`0.1pc/` … `150pc/`):  
Files are named `Earth_<O2>pc_<SZA>.txt` (e.g. `Earth_100pc_48.2.txt`). Each file contains column-integrated mixing ratios for the full chemical network (O₃, O, O₂, NOₓ, HOₓ, hydrocarbons, etc.) on a fixed altitude grid.

**`Young Sun/`** — faint young Sun simulations across the O₂ grid.

**`WACCM6_Sun_simulations/`** — runs using WACCM6 stellar spectra directly (`WACCMSun_<O2>pc_48.2.txt`).

**`Methane_Perturbations/`** — CH₄ boundary-condition sensitivity tests.

**`Albedo Perturbations/`** — surface albedo sensitivity at 0.1% and 100% PAL.

**`Proxima_Centauri/`** — Proxima Centauri b stellar input with Earth boundary conditions.

**`Old sims/`** — superseded early runs; see individual P–T profile sub-folders.

---

### `VULCAN/`

Configuration and input files for [VULCAN](https://github.com/shami-EEG/VULCAN) runs. VULCAN output files (`.vul`) are referenced in the Python analysis scripts but are not stored in this repository due to size.

**`config_files/`** — one configuration file per simulation. Naming convention:

```
cfg_Earth_<O2>pc_o2_1e12s_<SZA>SZA_WPT_1rtol.txt
```

Key settings in each config:

- `atm_file` — WACCM6 P–T and eddy-diffusion (Kzz) profile
- `sflux_file` — stellar UV spectrum (Gueymard solar or young-Sun files)
- `bot_BC_flux_file` — lower-boundary fluxes (CH₄, N₂O, etc.)
- `out_name` — corresponding `.vul` output filename

Methane perturbations are indicated in filenames (e.g. `0.01xCH4`, `10xCH4`). Water-vapour perturbations use the suffix `_0.6hum` (60% relative humidity scaling).

**`Inputs/`** — atmospheric profiles, stellar flux files, and lower-boundary condition tables shared across configs.

---

### `Kasting_1D_model/`

Output from the 1D photochemical model developed in the group headed by James Kasting ([Ji et al. 2024](https://zenodo.org/records/10822978)), which uses a correlated-*k* parameterisation for O₂ photolysis in the Schumann–Runge bands (175–205 nm).

**Standard grid** (`0.1pc/` … `150pc/`):  
Sub-folders for SZA (`SZA_45`, `SZA_48.2`, `SZA_60`) and an `8point` diurnal-integration option. Key output files:

- `outchem.dat` — chemical composition profiles
- `OUTPUT_PLOT.dat` — formatted output for plotting

**`WACCM_PT_Profiles/`** — WACCM6 P–T profiles used as input.

**`Ji 2024 fixed flux simulations with WACCM PT profiles/`** — runs with fixed lower-boundary fluxes and WACCM6 thermal structure across the full O₂ grid (0.1%–100% PAL).

**`Old sims/`** — legacy Kasting model setup files and early test output.

---

### `Python/`

Analysis and figure-generation code for the paper.

| File | Purpose |
|------|---------|
| `Early_Earth.py` | Main analysis script — loads model output, computes ozone columns, odd-oxygen budgets, and generates comparison figures. Run cell-by-cell in Spyder or as a script. |
| `early_earth_lib.py` | Shared library: I/O helpers, ozone column integrals, production/loss diagnostics, plotting utilities, and WACCM6 data readers. |

The scripts compare all four 1D models against WACCM6, including O₂–O₃ curves, catalytic loss budgets (NOₓ, HOₓ), methane and humidity perturbations, Proxima Centauri b cases, and latitude-resolved WACCM6 diagnostics.

> **Note:** Some hard-coded file paths in the Python scripts point to local WACCM6 NetCDF archives and VULCAN `.vul` output directories on the authors' machines. Update these paths before running locally.

---

### `PSG/`

Post-processing pipeline for generating synthetic planetary spectra via the [NASA Planetary Spectrum Generator (PSG)](https://psg.gsfc.nasa.gov/). The scripts convert 1D model output into PSG input files and submit them to the PSG API.

| File | Purpose |
|------|---------|
| `main.py` | Entry point — builds PSG configs and optionally uploads to PSG |
| `config.py` | Planet, star, geometry, and atmosphere file settings |
| `atmosphere.py` | Converts model output to PSG atmosphere format |
| `psg_writer.py` | Writes PSG configuration files |
| `plot_spectra.py` | Plots resulting transmission/emission spectra |

Used to assess how model and boundary-condition differences propagate into observables relevant to exoplanet characterisation.

---

## Key simulation suites

### Baseline O₂–O₃ grid

The core dataset spans eight O₂ levels (0.1%–150% PAL) × three solar zenith angles × four models, all using WACCM6 P–T profiles and present-day solar flux (`Sun_0.0Ga`). This grid maps the Proterozoic-to-Phanerozoic oxygenation history and forms the basis of the O₂–O₃ comparison curves in the paper.

### Faint young Sun

The `Young Sun/` (Photochem) and `Coupled Young Sun Simulations/` (Atmos) folders vary stellar age (e.g. 2.4 Ga) at low and high O₂, isolating the effect of a weaker, harder-UV young Sun on ozone production.

### Methane perturbations

CH₄ mixing ratio is scaled by factors from 0.01× to 10× present-day values. Methane affects HOₓ and odd-oxygen chemistry, particularly at low O₂, and is a major source of inter-model disagreement.

### Water vapour perturbations

Tropospheric H₂O is scaled (e.g. `_0.6hum` configs in VULCAN) to test sensitivity to atmospheric humidity, which influences UV shielding and HOₓ production.

### Surface albedo

Albedo is varied from 0.06 to 0.30 at fixed SZA to probe how surface reflectivity affects actinic flux and photolysis rates in the lower atmosphere.

### Proxima Centauri b

A parallel suite (`Proxima Centauri/` in Atmos, `Proxima_Centauri/` in Photochem) applies Earth-like boundary conditions with M-dwarf stellar flux and exoplanet P–T profiles, connecting the early-Earth work to exoplanet habitability.

---

## Citation

If you use this dataset or the associated models, please cite the paper (citation to be added upon publication) and the relevant model papers:

- **Kasting 1D / correlated-*k* O₂ photolysis:** Ji, A. et al. (2024), *A Correlated-k Parameterization for O₂ Photolysis in the Schumann-Runge Bands*, JGR Atmospheres. Data: [10.5281/zenodo.10822978](https://doi.org/10.5281/zenodo.10822978)
- **Photochem:** Wogan, N. F. et al. (2025), *The Open-source Photochem Code: A General Chemical and Climate Model for Interpreting (Exo)Planet Observations*, PSJ, 6, 256. [doi:10.3847/PSJ/ae0e1c](https://doi.org/10.3847/PSJ/ae0e1c)
- **VULCAN:** Tsai, S.-M. et al. (2017), *VULCAN: An Open-source, Validated Chemical Kinetics Python Code for Exoplanetary Atmospheres*, ApJS, 228, 20. [doi:10.3847/1538-4365/228/2/20](https://doi.org/10.3847/1538-4365/228/2/20)
- **Atmos:** Arney, G. et al. (2016), *The Pale Orange Dot: The Spectrum and Habitability of Hazy Archean Earth*, Astrobiology, 16, 873. [doi:10.1089/ast.2015.1422](https://doi.org/10.1089/ast.2015.1422)
- **WACCM6 / CESM:** A. Gettelman et al. (2019), The Whole Atmosphere Community Climate Model Version 6 (WACCM6), [https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JD030943](https://agupubs.onlinelibrary.wiley.com/doi/full/11.1029/2019JD030943), see [ESCOMP/CESM](https://github.com/ESCOMP/CESM)
- **Planetary Spectrum Generator (PSG):** Villanueva, G. L., et al.(2018), *Planetary Spectrum Generator: An accurate online radiative transfer suite for atmospheres, comets, small bodies and exoplanets*, Journal of Quantitative Spectroscopy and Radiative Transfer, 217, 86-104. [doi:10.1016/j.jqsrt.2018.05.023] (https://doi.org/10.1016/j.jqsrt.2018.05.023)

---

## Licence and contact

Simulation data and configuration files are provided to support reproducibility of the published results. For any questions, please contact either Akhil Kumar or Greg Cooke.
