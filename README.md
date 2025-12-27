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

PyCombustion-Solver implements a classical low-Mach, incompressible/reacting-flow formulation on a structured 2D Cartesian grid. The numerical methods are chosen to balance robustness, clarity, and pedagogical value, while remaining close to schemes used in production CFD codes.

### Governing equations and flow model

- **Momentum**: 2D incompressible Navier–Stokes equations with variable density, including viscous diffusion and buoyancy forcing.
- **Scalars**: Convection–diffusion–reaction equations for temperature and species (fuel, oxidizer, products, water, NOx).
- **Incompressibility constraint**: A pressure-projection step enforces a divergence-free velocity field, which is standard for low-Mach reacting flows where acoustics are filtered out.

### Spatial discretization

- **Finite-volume formulation**: All variables are stored at cell centers on a uniform grid; fluxes are evaluated at cell faces.
- **Convection**: MUSCL/TVD schemes (Minmod, Superbee, Van Leer, etc.) are used to reconstruct face values. These schemes are:
   - formally second-order accurate in smooth regions, and
   - total-variation-diminishing (TVD), which suppresses spurious oscillations near steep gradients while keeping numerical diffusion under control.
- **Diffusion and pressure**: Second-order centered differences are used for viscous and thermal diffusion and for the pressure–Poisson operator. This choice keeps the stencil simple and symmetric, which is helpful for multigrid solvers and for students reading the code.

### Time integration and stability

- **Explicit time stepping**: The solver uses first-order explicit (forward Euler) time integration for momentum, scalars, and chemistry source terms.
- **CFL-based time step control**: The time step is computed from separate convective and diffusive CFL limits. Although first-order explicit schemes are only conditionally stable, enforcing these CFL criteria makes the method robust and easy to reason about in an educational setting.
- **Chemistry subcycling**: Stiff chemical source terms are integrated with several smaller substeps within each global time step. This keeps the implementation simple (no external ODE solvers) while maintaining reasonable stability for typical laminar/diffusion-flame test cases.

### Pressure projection and Poisson solver

- **Projection method**: After computing an intermediate velocity without pressure, a variable-coefficient Poisson equation is solved for the pressure correction. The corrected velocity then satisfies the incompressibility constraint.
- **Multigrid V-cycle**: The Poisson problem is solved using a geometric multigrid V-cycle implemented in `CBm0/numerics.py`. Multigrid is chosen because it is conceptually simple yet demonstrates how scalable pressure solvers are built in larger CFD codes.

### Chemistry, thermodynamics, and transport

- **Global reaction mechanisms**: The methane (CH4) two-step Westbrook–Dryer mechanism and a one-step global hydrogen (H2) mechanism are used. These global models capture key flame behavior at a fraction of the cost of detailed chemistry, making them well-suited for interactive, Python-based simulations.
- **Thermodynamics**: Mixture properties are computed from Shomate polynomials for species heat capacities and enthalpies. This allows realistic temperature dependence without relying on external databases.
- **Transport**: Mixture viscosity and species diffusion coefficients follow Wilke-type mixture rules, with options for simplified unity-Lewis-number transport. These models are common in low-Mach combustion solvers and keep the implementation compact.

### Code organization

- `CBm0/main.py`: Main solver loop (time stepping, boundary conditions, projection, chemistry subcycling, diagnostics/output).
- `CBm0/numerics.py`: Finite-volume operators (TVD/MUSCL advection, diffusion, divergence, multigrid Poisson solver).
- `CBm0/physics.py`: Thermodynamic and transport models (Shomate cp/h, mixture gas constant and density, viscosity and diffusion, optional thermo lookup tables, optional Le = 1 transport).
- `CBm0/chemistry.py`: Reaction source terms (CH4 two-step Westbrook–Dryer, global H2 one-step, optional thermal NO source).
- `CBm0/config.py`: Simulation settings (grid, inlets, numerics/CFL, chemistry, transport/thermo presets, output options).
- `CBm0/io_utils.py`: Output utilities (directory creation, contour plots, log-scale plots, run_config snapshots, plotting ranges).

Temperature and heat-release-rate contours in the example cases are often drawn on a logarithmic scale. Very small positive thresholds are applied to avoid numerical issues when taking the logarithm of near-zero fields.
Simulation is conducted under initial conditions of v_air=0.2m/s, v_fuel=0.6m/s with CH4 combustion setting.

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/245977a8-2625-48ae-bacf-6bebabc9335c" width="100%" />
      <br />
      <b>Figure 1: Temperature Contour
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/faf3561b-22bb-4911-b2e0-b0cd0daedd26" width="100%" />
      <br />
      <b>Figure 2: Heat Realease Rate Contour
    </td>
  </tr>
</table>


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff).































