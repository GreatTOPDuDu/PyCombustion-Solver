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

Clone the repository and install the package:

```bash
git clone https://github.com/your-id/PyCombustion-Solver.git
cd PyCombustion-Solver
pip install -e .
```

## Quick Start

You can run the included demo script to see the solver in action:

```bash
python examples/run_demo.py
```

Or run the solver module directly with a configuration file:

```bash
python -m CBm0 --config examples/demo_config.yaml
```

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





