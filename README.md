# PyCombustion-Solver (CBm0)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python](https://img.shields.io/badge/Python-3.9_to_3.11-blue)
[![DOI](https://zenodo.org/badge/1105952822.svg)](https://doi.org/10.5281/zenodo.17750354)
[![Python package](https://github.com/GreatTOPDuDu/PyCombustion-Solver/actions/workflows/python-package.yml/badge.svg)](https://github.com/GreatTOPDuDu/PyCombustion-Solver/actions/workflows/python-package.yml)

**PyCombustion-Solver** (internal module name `CBm0`) is a compact, educational 2D reacting flow solver written in Python. It is designed to bridge the gap between textbook theory and complex commercial CFD packages.

## Features

- **2D Incompressible Flow**: Navier-Stokes equations with pressure projection (Multigrid V-cycle).
- **Convection Schemes**: MUSCL/TVD (Minmod, Superbee, etc.) for robust shock/gradient handling.
- **Chemistry**:
  - Methane (CH4) 2-step Westbrook-Dryer mechanism.
  - Hydrogen (H2) 1-step global mechanism.
  - Thermal NOx approximation.
- **Thermodynamics**: Temperature-dependent specific heat (Shomate equations) and enthalpy.
- **Transport**: Wilke's mixture viscosity model.
- **Performance**: Accelerated with Numba (JIT compilation).

## Installation

It is recommended to use a virtual environment to avoid conflicts. Also recommend to use code in Window settings.

Clone the repository and install the package:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GreatTOPDuDu/PyCombustion-Solver.git
   cd PyCombustion-Solver
   ```

2. **Create and activate a virtual environment (Recommended):**
    *   **Windows (PowerShell):**
          ```powershell
          py -3.11 -m venv .venv
          .\.venv\Scripts\Activate
          ```
    *   **Windows (Command Prompt / cmd):**
          ```cmd
          py -3.11 -m venv .venv
          .\.venv\Scripts\activate
          ```
    *   **Linux/macOS (bash/zsh):**
          ```bash
          python3 -m venv .venv
          source .venv/bin/activate
          ```

3. **Install the package:**
    ```bash
    pip install -e .

## Quick Start

You can run the included demo script to see the solver in action:

```bash
py examples/run_demo.py
```

## Usage by Environment

Below are step-by-step instructions for common environments.

### Windows (PowerShell)
- **Run demo:**
   ```powershell
   py examples/run_demo.py
   ```

### Windows (Command Prompt / cmd)
- **Run demo:**
   ```cmd
   py examples\run_demo.py
   ```

### Linux/macOS (Terminal)
- **Run demo:**
   ```bash
   python3 examples/run_demo.py
   ```

### Notes
- If `py` is unavailable on Linux/macOS, use `python3`.
- Numba will JIT-compile on first run; the initial execution may be slower.
- Outputs are written under `outputs/demo_result/` when running the demo.

## Pytest (If required)

```bash
python -m pip install pytest
python -m pytest
```

## Documentation

The code is structured as follows:
- `CBm0/main.py`: Main solver loop (time stepping, BCs, projection, chemistry subcycling, diagnostics/output).
- `CBm0/numerics.py`: Finite-volume operators (TVD/MUSCL advection, diffusion, divergence, multigrid Poisson solver).
- `CBm0/physics.py`: Thermodynamic & transport models (Shomate cp/h, mixture gas constant and density, viscosity and diffusion, optional thermo LUT, optional Le=1 transport).
- `CBm0/chemistry.py`: Reaction source terms (CH4 two-step Westbrook–Dryer, global H2 one-step, optional thermal NO source).
- `CBm0/config.py`: Simulation settings (grid, inlets, numerics/CFL, chemistry, transport/thermo presets, output options).
- `CBm0/io_utils.py`: Output utilities (directory creation, contour plots, log-scale plots, run_config snapshot, plotting ranges).

### Governing variables and modeling assumptions

**Primary fields**
- Velocity: `u(x,y,t)`, `v(x,y,t)`
- Temperature: `T(x,y,t)`
- Species mass fractions:
  - Fuel: `YF` (CH4 or H2, selected by `fuel_type`)
  - Oxidizer: `YO` (O2)
  - Products/intermediates: `YC` (CO), `YC2` (CO2), `YW` (H2O)
  - Pollutant tracer: `YNO` (NO)

**Mixture closure**
- `YN2 = 1 - (YF + YO + YC + YC2 + YW)` is reconstructed internally for mixture properties.
- Density is computed from an ideal-gas mixture at a fixed reference pressure `P0_pa`:
  - `rho = P0 / (Rmix * T)`
- For mixture properties (cp, `Rmix`, viscosity, diffusion), `YNO` is treated as a sparse tracer and is not included in the closure sum.

**Buoyancy (simple plume effect)**
- A vertical buoyancy acceleration is added to the `v`-momentum predictor:
  - `b = g * (rho_ref - rho) / rho_ref`
  - `rho_ref` is computed from the air reference composition and `T_air`.

---

### Chemistry models and heat release rate (HRR)

**CH4 option (`kinetics_model="wd2"`, `fuel_type="CH4"`)**
- Two-step Westbrook–Dryer (global) mechanism:
  1) CH4 + 1.5 O2 -> CO + 2 H2O
  2) CO + 0.5 O2 -> CO2
- HRR is computed using temperature-dependent Shomate molar enthalpies `h(T)`:
  - `HRR = - sum_over_reactions( rate_r * dH_r(T) )`
- Global sensitivity knobs: `wd2_A_mult`, `wd2_Ea_mult`.

**H2 option (`kinetics_model="h2"`, `fuel_type="H2"`)**
- One-step global model: H2 + 0.5 O2 -> H2O with a simple Arrhenius-type rate.
- Global sensitivity knobs: `h2_A_mult`, `h2_Ea_mult`.

**Thermal NO (optional)**
- If `enable_thermal_NOx=True`, a simplified thermal-NO source term is added to the `YNO` equation (intended as a lightweight indicator).

**Chemistry coupling**
- Reaction source terms are integrated with subcycling each flow time step:
  - `CHEM_SUBSTEPS` substeps with `dt_sub = dt / CHEM_SUBSTEPS`.

---

### Numerical method (high-level)

**Spatial discretization**
- Uniform Cartesian mesh:
  - `Nx`, `Ny` points over `[0, Lx] x [0, Ly]`
  - `dx = Lx/(Nx-1)`, `dy = Ly/(Ny-1)`

**Advection (scalars and momentum)**
- TVD/MUSCL-type reconstruction with selectable limiter:
  - `adv_limiter` in `{minmod, vanleer, superbee}`
- Implemented by `numerics.tvd_div()` for transported fields.

**Diffusion**
- Explicit diffusion operator `numerics.apply_diffusion()` with field-dependent coefficients:
  - Temperature: thermal diffusivity `alpha`
  - Species: Fickian diffusion `D_i(T)`
  - Momentum: kinematic viscosity `nu = mu / rho`

**Time stepping**
- Explicit update with stable time step selected from:
  - Advection CFL: `cfl_adv`
  - Diffusion CFL: `cfl_diff`
- Chemistry is integrated separately via subcycling.

**Pressure-velocity coupling (projection)**
- Predictor step forms `(u*, v*)` with advection + diffusion + buoyancy.
- Pressure correction is obtained by solving a variable-coefficient Poisson-type equation using a multigrid V-cycle:
  - Smoother: red-black Gauss–Seidel (Numba JIT + `parallel=True`)
  - Restriction: simple averaging (Numba JIT + `parallel=True`)
  - Prolongation: bilinear
- Corrected velocity update (conceptually):
  - `u^{n+1} = u* - dt * k * grad(p')`, with `k = 1/rho`

---

### Boundary conditions and inlet specification

**Coordinate convention**
- `x` is horizontal, `y` is vertical.
- Inlet: `y=0` (bottom boundary). Outlet: `y=Ly` (top boundary).

**Scalars (T and species)**
- Inlet (`y=0`): Dirichlet from inlet profiles (`T_inlet`, `YF_inlet`, `YO_inlet`), other species set to 0 at inlet.
- Side walls (`x=0`, `x=Lx`): zero-gradient (copy from adjacent interior cell).
- Outlet (`y=Ly`): convective outflow update with an estimated outflow speed.

**Velocity**
- Inlet (`y=0`): `u=0`, `v=v_inlet` (vertical injection).
- Side walls (`x=0`, `x=Lx`): no-slip (`u=0`, `v=0`).
- Outlet (`y=Ly`): convective outflow update.

**Inlet mixing modes**
- `mixing_mode="stratified"`:
  - Fuel enters through one or more inlet "channels", while the rest is air.
  - Channel pattern can be created by:
    - `inlet_mode="uniform"` with `num_fuel_channels` and `fuel_channel_width`
    - `inlet_mode="explicit"` with `fuel_inlet_spans_m` (x-spans in meters)
- `mixing_mode="premixed"`:
  - A premixed stream is synthesized using an equivalence ratio choice:
    - `equiv_mode` in `{rich, stoic, lean}` or `phi_override`
  - Note: current premix stoichiometry constant is set for CH4.

---

### Configuration presets (reference / fast / inert)

The single switch `mode` controls high-level fidelity:

- `mode="reference"`
  - Analytic Shomate cp/h evaluation every step
  - Full transport model (Wilke mixture viscosity + species diffusion)
  - Chemistry enabled
- `mode="fast"`
  - Thermo cp LUT interpolation (`use_thermo_lut=True`)
  - Simplified Le=1 transport (`transport_model="Le1"`)
  - Chemistry enabled
- `mode="inert"`
  - Chemistry disabled (`chemistry_on=False`)
  - Thermo/transport remain as configured

---

### Outputs and how to interpret the figures

**Directory layout**
- Output root: `output_base_dir/output_method_dir/`
- Subfolders are created automatically, for example:
  - `Temperature/`, `FuelFraction/`, `O2Fraction/`, `H2OFraction/`
  - `HeatReleaseRate/` (log-scale plots)
  - `Density/`, `Pressure/`, `u_velocity/`, `v_velocity/`
  - `NOFraction/` (log-scale plots)
  - `COFraction/`, `CO2Fraction/` (created only for `kinetics_model="wd2"`)
  - `diagnostics/`

**What gets written**
- `run_config.txt`: full configuration snapshot saved at the beginning of the run.
- `diagnostics/diagnostics.csv`: per-step summary including:
  - integrated HRR metrics (domain-integrated, per-unit-depth style)
  - cumulative chemical energy integral
  - `Tmax`, `Tmean`, density stats, and mass-fraction sum checks
- Optional monitoring (if enabled):
  - `diagnostics/monitor.log`
  - `diagnostics/centerline_T.txt` (centerline temperature history blocks)

**Plot scaling**
- Linear fields (e.g., Temperature) use fixed min/max if provided in config (`plot_T_min`, `plot_T_max`, etc.), otherwise auto-scale.
- Log fields (e.g., HRR, CO, NO) use `LogNorm` with config-driven minima (`plot_HRR_min`, `plot_YC_min`, `plot_YNO_min`, etc.).

**Relating to the example images**
- The first image is a temperature contour.
- The second image is an HRR contour drawn on a log scale; near-zero values are clipped to a small positive threshold to remain compatible with logarithmic plotting.

---

### Reproducing the example run

1) Edit parameters in `CBm0/config.py` (the `Config` dataclass defaults).
2) Run the solver:

```bash
python -m CBm0


![(0 6,0 2)_t=1 5_core8_thick](https://github.com/user-attachments/assets/245977a8-2625-48ae-bacf-6bebabc9335c) ![(0 6,0 2)_t=1 5_core8_thick_HRR](https://github.com/user-attachments/assets/faf3561b-22bb-4911-b2e0-b0cd0daedd26)


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff).























