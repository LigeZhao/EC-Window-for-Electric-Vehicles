# EC-Window-for-Electric-Vehicles
# Smart Window Co-simulation Framework for EV Thermal Management

**Code for:** *Climate-dependent thermal regulation by electrochromic glazing extends electric vehicle driving range across global cities*
<!-- DOI badge will be added here after publication -->

This repository contains the simulation framework used in the above study. The framework couples a multi-node cabin thermal model with an electric vehicle (EV) thermal management system (TMS) and FASTSim-based powertrain simulation to evaluate the impact of smart (electrochromic) window glazing materials on cabin thermal comfort, TMS energy consumption, and driving range across global climate conditions.

---

## Repository Structure

```
├── thermal_functions.py      # Physical model library: Environment, Cabin, Battery,
│                             #   Motor, Compressor, Condenser, Evaporator, Radiator, PID
├── TMS_operation.py          # TMS operational modes: AC, Heat Pump, Ventilation
├── co_simulation.py          # Main co-simulation loop (FASTSim + TMS coupling)
├── run_simulation.py         # Batch simulation runner (multi-core, CLI)
├── plot_results.py           # Publication figure generation (Fig. 1–11)
├── plot_conf_figures.py      # Conference figure generation
├── download_epw.py           # Automated EPW weather data downloader (NREL)
├── weather_upload.py         # EPW file parser
├── glass_materials.json      # Glass material parameter database (8 types)
├── cycles/                   # Driving cycle definitions (WLTP, CLTC, UDDS, HWFET, ...)
└── docs/
    └── FASTSim_py_veh_db.csv # Vehicle parameter database (from NREL FASTSim)
```

---

## Installation

It is recommended to use a Conda environment:

```bash
conda env create -f environment.yml
conda activate smart-window
```

Or install with pip:

```bash
pip install -r requirements.txt
```

> **Note:** `cartopy` may require system-level libraries (GEOS, PROJ). If installation fails, follow the [Cartopy installation guide](https://scitools.org.uk/cartopy/docs/latest/installing.html) or install via conda-forge (recommended).

---

## Weather Data

Weather data (EnergyPlus EPW files) are not included in this repository. Download them using the provided script, which fetches files from the [NREL EnergyPlus Weather Data](https://energyplus.net/weather) repository:

```bash
# Download EPW files for all available global stations (into ./weather_data/)
python download_epw.py --output-dir ./weather_data

# Download for a specific region only (e.g., Asia)
python download_epw.py --output-dir ./weather_data --region Asia
```

---

## Running a Simulation

### Single location (programmatic)

```python
from co_simulation import sim_drive, get_standard_cycle, get_veh
from weather_upload import get_weather

# Load inputs
cycle = get_standard_cycle('wltp')
veh   = get_veh(vnum=9)          # Vehicle #9 from FASTSim database (Tesla Model 3)
weather = get_weather('weather_data/CHN_Beijing.epw', month=7)

# Run
result = sim_drive(cycle, veh, weather, glass_scheme='normal_glass_with_ec_colored')
```

### Batch simulation (multi-core CLI)

```bash
python run_simulation.py \
    --epw-dir ./weather_data \
    --output-dir ./simulation_result \
    --jobs 8 \
    --glass normal_glass tinted_glass normal_glass_with_ec_colored
```

Run `python run_simulation.py --help` for all options, including `--months`, `--headings`, and `--resume`.

---

## Glass Material Schemes

The following glazing configurations are defined in `glass_materials.json`:

| Scheme key | Description |
|---|---|
| `normal_glass` | Standard soda-lime glass (baseline) |
| `tinted_glass` | Tinted glass on side/rear windows |
| `high_trans_glass` | High-transmittance low-E glass |
| `normal_glass_with_ec_trans` | Normal glass + electrochromic (EC) film, clear state |
| `normal_glass_with_ec_colored` | Normal glass + EC film, colored state (windshield τ_vis ≥ 0.70) |

---

## Driving Cycles

| File | Standard | Region |
|---|---|---|
| `wltp.csv` | WLTP | Global / EU |
| `double_wltp.csv` | WLTP × 2 | Extended range test |
| `cltc.csv` | CLTC | China |
| `udds.csv` | UDDS | USA (urban) |
| `double_udds.csv` | UDDS × 2 | Extended |
| `hwfet.csv` | HWFET | USA (highway) |

---

## Reproducing Paper Figures

After running the batch simulation, generate all paper figures with:

```bash
python plot_results.py --result-dir ./simulation_result
```

Individual figures can be called programmatically:

```python
from plot_results import plot_fig1, plot_fig2  # ... plot_fig11
plot_fig1()   # e.g., annual energy breakdown by glass scheme
```

---

## Third-Party Acknowledgements

- **NREL FASTSim** — Vehicle powertrain simulation model and vehicle database (`docs/FASTSim_py_veh_db.csv`).
  Brady, N., Hendricks, T., Lustbader, J. et al. *FASTSim: A Model to Estimate Vehicle Efficiency, Cost, and Performance*. NREL. https://github.com/NREL/fastsim

- **CoolProp** — Thermophysical property library for refrigerant calculations.
  Bell, I.H. et al. *Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp*. Industrial & Engineering Chemistry Research, 53(6), 2498–2508, 2014.

- **EnergyPlus Weather Data** — EPW climate files distributed by the U.S. Department of Energy / NREL.
  https://energyplus.net/weather

---

## Citation

> *[Full citation will be added upon publication.]*

If you use this code before the paper is published, please cite this repository:

```
Zhao, L. (2026). Climate-dependent thermal regulation by electrochromic glazing extends electric vehicle driving range across global cities.
https://github.com/LigeZhao/EC-Window-for-Electric-Vehicles
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

© 2026 Lige Zhao, Smart Building Lab, The Hong Kong University of Science and Technology (HKUST)
