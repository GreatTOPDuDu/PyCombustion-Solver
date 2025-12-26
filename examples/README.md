# Examples

This directory collects small scripts and configuration files that exercise
`PyCombustion-Solver`.

- `run_demo.py`, `demo_config.yaml`, `fast_demo_config.yaml`:
  Buoyant diffusion flame / plume demonstration cases used for the figures
  in the paper.
- `run_cavity.py`, `cavity_config.yaml`, `cavity_reference/`,
  `postprocess_cavity.py`:
  Lid-driven cavity validation at Re=100 (Ghia et al.).
- `run_advection_1d.py`, `advection_config.yaml`, `postprocess_advection.py`:
  1D top-hat scalar advection tests for MUSCL/TVD limiters.
- `run_ignition_0d.py`, `ignition_0d_config.yaml`, `postprocess_ignition_0d.py`:
  Zero-dimensional ignition test for the global reaction mechanism.

The `*_output/` and `validation_figs/` folders are generated at runtime and
are excluded from version control via the top-level `.gitignore`.
