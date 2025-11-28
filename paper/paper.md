---
title: 'PyCombustion-Solver (CBm0): An Educational 2D Reacting Flow Solver in Python'
tags:
  - Python
  - CFD
  - combustion
  - fluid dynamics
  - numba
  - reacting flows
authors:
  - name: J. H. Song
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 28 November 2025
bibliography: paper.bib
---

# Summary

`PyCombustion-Solver` (internal module name `CBm0`) is a Python-based solver designed for simulating two-dimensional, incompressible, reacting flows. It is built upon `NumPy` and accelerated with `Numba` to provide a balance between code readability and computational performance. The solver includes modules for finite-volume discretization, multi-grid pressure projection, and simplified chemical kinetics (e.g., Westbrook-Dryer mechanism for methane).

# Statement of need

Computational Fluid Dynamics (CFD) of reacting flows is a complex field often dominated by large, monolithic software packages like ANSYS Fluent or OpenFOAM. While these tools are powerful, their steep learning curve and complex codebases can be barriers for students and researchers who wish to understand the fundamental coupling between fluid mechanics, thermodynamics, and chemical kinetics.

`PyCombustion-Solver` addresses this need by providing a compact, transparent implementation of a reacting flow solver. It allows users to inspect and modify every part of the pipeline—from the advection schemes (MUSCL/TVD) to the pressure solver (Multigrid) and chemical source term integration. It serves as an educational tool for graduate courses in combustion and CFD, as well as a prototyping platform for testing new numerical ideas or simplified reaction mechanisms before implementation in larger codes.

# Functionality

The solver features:
- **Incompressible Navier-Stokes Solver**: Uses a projection method with a V-cycle multigrid Poisson solver for pressure.
- **Transport**: Handles variable density and temperature-dependent transport properties (viscosity, thermal conductivity) using Wilke's mixing rule [@Wilke1950].
- **Thermodynamics**: Calculates specific heat and enthalpy using NASA Shomate polynomials [@ShomateNIST].
- **Chemistry**: Supports global reaction mechanisms, including the 2-step Westbrook-Dryer mechanism for methane [@WestbrookDryer1981] and a 1-step hydrogen mechanism.
- **Numerics**: Implements second-order TVD schemes (Minmod, Superbee, etc.) for advection to minimize numerical diffusion while maintaining stability [@VersteegMalalasekera2007].

# Validation

The solver has been validated against standard benchmarks, including:
- **Lid-driven cavity flow**: To verify the pressure projection and viscous terms.
- **Scalar advection tests**: To assess the performance of TVD limiters.
- **0D Ignition delays**: Comparing the integrated chemistry against canonical Arrhenius rates.

# References
