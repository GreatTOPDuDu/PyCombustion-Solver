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
- `CBm0/main.py`: Main solver loop.
- `CBm0/numerics.py`: Finite volume schemes (fluxes, pressure solver).
- `CBm0/physics.py`: Thermodynamic and transport properties.
- `CBm0/chemistry.py`: Chemical reaction source terms.
- `CBm0/config.py`: Simulation settings.
- 'CBm0/io_utils.py': Simulation result visualization.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff).


















