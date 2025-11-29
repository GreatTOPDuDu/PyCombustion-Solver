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

Or run the solver module directly with a configuration file:

```bash
py -m CBm0 --config examples/demo_config.yaml
```

## Usage by Environment

Below are step-by-step instructions for common environments.

### Windows (PowerShell)
- **Create venv:**
   ```powershell
   py -3.11 -m venv .venv
   ```
- **Activate venv:**
   ```powershell
   .\.venv\Scripts\Activate
   ```
- **Install in editable mode:**
   ```powershell
   pip install -e .
   ```
- **Run demo:**
   ```powershell
   py examples/run_demo.py
   ```
- **Run with config:**
   ```powershell
   py -m CBm0 --config examples/demo_config.yaml
   ```

### Windows (Command Prompt / cmd)
- **Create venv:**
   ```cmd
   py -3.11 -m venv .venv
   ```
- **Activate venv:**
   ```cmd
   .\.venv\Scripts\activate
   ```
- **Install in editable mode:**
   ```cmd
   pip install -e .
   ```
- **Run demo:**
   ```cmd
   py examples\run_demo.py
   ```
- **Run with config:**
   ```cmd
   py -m CBm0 --config examples\demo_config.yaml
   ```

### Windows (VS Code)
- **Open folder:** Start VS Code and open the repository folder.
- **Select Python interpreter:** Use `Ctrl+Shift+P` → `Python: Select Interpreter` → choose `.venv` interpreter.
- **Create venv (optional if not created yet):** Open the integrated terminal (`PowerShell` by default) and run:
   ```powershell
   py -3.11 -m venv .venv; .\.venv\Scripts\Activate; pip install -e .
   ```
- **Run demo:**
   - From terminal:
      ```powershell
      py examples/run_demo.py
      ```
   - Or press `F5` to run a configured launch (you can add a simple `launch.json` to run `examples/run_demo.py` if desired).
- **Run with config:**
   ```powershell
   py -m CBm0 --config examples/demo_config.yaml
   ```

### Linux/macOS (Terminal)
- **Create venv:**
   ```bash
   python3 -m venv .venv
   ```
- **Activate venv:**
   ```bash
   source .venv/bin/activate
   ```
- **Install in editable mode:**
   ```bash
   pip install -e .
   ```
- **Run demo:**
   ```bash
   python3 examples/run_demo.py
   ```
- **Run with config:**
   ```bash
   python3 -m CBm0 --config examples/demo_config.yaml
   ```

### Notes
- If `py` is unavailable on Linux/macOS, use `python3`.
- Numba will JIT-compile on first run; the initial execution may be slower.
- Outputs are written under `outputs/demo_result/` when running the demo.

## Documentation

The code is structured as follows:
- `CBm0/main.py`: Main solver loop.
- `CBm0/numerics.py`: Finite volume schemes (fluxes, pressure solver).
- `CBm0/physics.py`: Thermodynamic and transport properties.
- `CBm0/chemistry.py`: Chemical reaction source terms.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff).










