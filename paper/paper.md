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

`PyCombustion-Solver` (internal module name `CBm0`) is a compact, Python-based solver for two-dimensional, low-Mach, incompressible reacting flows. It is built on `NumPy` and accelerated with `Numba` to balance code readability with reasonable performance on desktop hardware. The solver implements a finite-volume formulation with MUSCL/TVD advection schemes, a pressure-projection step solved by geometric multigrid, and global reaction mechanisms for methane and hydrogen based on the Westbrook–Dryer family of models [@WestbrookDryer1981]. The design goal is to expose, in a single, readable code base, the numerical building blocks that appear in much larger CFD and combustion codes: spatial reconstruction, CFL-limited time integration, pressure Poisson solves, thermodynamic property evaluation, and stiff chemical source term handling.

# Statement of need

Computational Fluid Dynamics (CFD) of reacting flows is a complex field often dominated by large, monolithic software packages like ANSYS Fluent or OpenFOAM. While these tools are powerful, their steep learning curve and complex codebases can be barriers for students and researchers who wish to understand the fundamental coupling between fluid mechanics, thermodynamics, and chemical kinetics.

`PyCombustion-Solver` addresses this need by providing a compact, transparent implementation of a reacting flow solver. It allows users to inspect and modify every part of the pipeline—from the advection schemes (MUSCL/TVD) to the pressure solver (Multigrid) and chemical source term integration. It serves as an educational tool for graduate courses in combustion and CFD, as well as a prototyping platform for testing new numerical ideas or simplified reaction mechanisms before implementation in larger codes.

# Functionality

The solver follows a classical low-Mach reacting-flow formulation on a structured 2D Cartesian grid.

- **Flow and scalar equations**: The code advances the incompressible Navier–Stokes equations with variable density, along with convection–diffusion–reaction equations for temperature and multiple species (fuel, oxidizer, products, water, NOx). A projection method enforces a divergence-free velocity field by solving a variable-coefficient Poisson equation for pressure.
- **Spatial discretization**: A cell-centered finite-volume formulation is used. Convective fluxes are computed with MUSCL/TVD reconstructions (Minmod, Superbee, Van Leer, etc.), which are second-order accurate in smooth regions and total-variation-diminishing (TVD) near steep gradients [@VersteegMalalasekera2007]. Diffusive and pressure terms use second-order centered differences, yielding a symmetric stencil that is easy to understand and efficient for multigrid solvers.
- **Time integration and stability control**: All transport equations are integrated explicitly with first-order forward Euler. The global time step is chosen from separate convective and diffusive CFL criteria, making the stability limitations of explicit schemes transparent to students. Chemical source terms are subcycled within each global time step, using smaller internal steps to integrate stiff reactions while keeping the implementation free from external ODE libraries.
- **Pressure projection and multigrid**: The pressure–Poisson problem arising from the projection step is solved by a geometric multigrid V-cycle with simple relaxation smoothers. This demonstrates how scalable pressure solvers are constructed in larger CFD codes, while remaining compact enough to inspect in a single source file.
- **Chemistry, thermodynamics, and transport**: Global reaction mechanisms are provided for methane (two-step Westbrook–Dryer) and hydrogen (one-step global). Thermodynamic properties are obtained from NASA Shomate polynomials [@ShomateNIST], and mixture-averaged viscosity and diffusivities are computed using Wilke-type mixing rules [@Wilke1950]. These choices offer physically reasonable behaviour at modest cost, making the examples suitable for interactive exploration in a teaching context.
- **Implementation**: The hotspot numerical kernels (advection, diffusion, multigrid operators, chemistry) are decorated with `Numba` JIT compilation, so that users can work in pure Python while still running simulations of practical size on a laptop or workstation.

# Governing equations and numerical schemes

PyCombustion-Solver advances a standard set of low-Mach reacting-flow equations for velocity $\boldsymbol{u}=(u,v)$, pressure $p$, temperature $T$, and species mass fractions $Y_k$ on a two-dimensional domain.

- **Continuity (incompressible constraint)**

$$
\nabla \cdot \boldsymbol{u} = 0
$$

- **Momentum equations**

$$
\frac{\partial \boldsymbol{u}}{\partial t} + \nabla \cdot (\boldsymbol{u}\boldsymbol{u}) = -\frac{1}{\rho}\nabla p + \nabla \cdot (\nu \nabla \boldsymbol{u})  + \boldsymbol{g}_\text{buoy}
$$

  where $\rho(T, Y_k)$ is the mixture density, $\nu$ is the kinematic viscosity, and $\boldsymbol{g}_\text{buoy}$ represents buoyancy forces.

- **Scalar transport (temperature and species)**

$$
\frac{\partial \phi}{\partial t}
+ \nabla \cdot (\boldsymbol{u}\,\phi)
= \nabla \cdot (D_\phi \nabla \phi)
  + \dot{\omega}_\phi
$$

  with $\phi \in \{T, Y_k\}$, effective diffusivity $D_\phi$, and chemical source term $\dot{\omega}_\phi$ obtained from the global reaction mechanisms.

These equations are discretized in a cell-centred finite-volume form on a uniform Cartesian grid. Fluxes across each cell face are computed as follows.

- **Convective fluxes**: Face values are reconstructed using MUSCL/TVD schemes with slope limiters (Minmod, Superbee, Van Leer). In smooth regions this yields second-order accurate upwind fluxes; near sharp gradients the limiter reduces to a first-order monotone scheme to preserve stability.
- **Diffusive fluxes and pressure operator**: Diffusive terms and the pressure–Poisson operator use second-order centred differences, which produce a symmetric stencil suitable for geometric multigrid.

Time integration employs first-order explicit (forward Euler) updates for all variables. The time step $\Delta t$ is restricted by separate convective and diffusive CFL conditions,

$$
\Delta t_\text{adv} =
\frac{C_\text{adv}}{|u|/\Delta x + |v|/\Delta y},
\qquad
\Delta t_\text{diff} =
\frac{C_\text{diff}}{\kappa (1/\Delta x^2 + 1/\Delta y^2)}
$$

and the solver uses $\Delta t = \min(\Delta t_\text{adv}, \Delta t_\text{diff})$. Here $C_\text{adv}$ and $C_\text{diff}$ are user-configurable CFL numbers and $\kappa$ represents the largest relevant diffusion coefficient (thermal, viscous, or species).

At each time step, an intermediate velocity $\boldsymbol{u}^*$ is first obtained without pressure. Enforcing $\nabla\cdot\boldsymbol{u}^{n+1}=0$ then leads to the pressure–Poisson equation

$$
\nabla \cdot \left( \frac{1}{\rho} \nabla p' \right)
  = \frac{1}{\Delta t} \nabla \cdot \boldsymbol{u}^*
$$

which is solved by a multigrid V-cycle. The corrected velocity is updated as

$$
\boldsymbol{u}^{n+1} = \boldsymbol{u}^* - \Delta t \; \frac{1}{\rho} \nabla p'
$$

Chemical source terms $\dot{\omega}_\phi$ are integrated explicitly using substepping within each global time step, keeping the implementation simple while allowing users to explore the interaction between transport and reaction timescales.

# Usage and reproducibility

The software is distributed as a Python package with all example cases and configuration files version-controlled alongside the source. A typical workflow is:

1. Clone the repository and install the package in editable mode (within a virtual environment):

  ```bash
  git clone https://github.com/GreatTOPDuDu/PyCombustion-Solver.git
  cd PyCombustion-Solver
  pip install -e .
  ```

2. Run one of the bundled examples, for instance the buoyant diffusion flame demo:

  ```bash
  py examples/run_demo.py
  ```

This script reads a YAML configuration file, advances the solution in time, and writes contour plots and numerical output (e.g., centerline temperature profiles) to an output directory. All figures shown in this paper are generated directly from such example scripts and configuration files, so that reviewers and readers can reproduce them by running the same commands on their own machines.

# Validation

The example cases bundled with the code are designed to exercise the main physical and numerical components of the solver against standard benchmarks from the CFD and combustion literature. All validation figures are generated by scripts in the `examples/` directory and written to `examples/validation_figs/`.

- **Lid-driven cavity flow (incompressible flow and projection)**: A 2D lid-driven cavity configuration at $Re=100$ is used to check the viscous terms and pressure projection. The case is run on a $129\times129$ uniform grid with a unit-square domain, no-slip walls, and a unit-velocity lid at the top wall. Centerline velocity profiles at the final time are compared to the tabulated 129-grid solution of Ghia et al. [@Ghia1982]. Using `examples/run_cavity.py` and `examples/postprocess_cavity.py`, the maximum relative errors in the vertical and horizontal centreline velocities are found to be approximately $4.6\%$ for $u(y)$ at $x=0.5$ and $15.3\%$ for $v(x)$ at $y=0.5$. The corresponding plots are saved as `cavity_u_centerline.png` and `cavity_v_centerline.png`.
- **Scalar advection tests (MUSCL/TVD behaviour)**: One-dimensional passive scalar advection with a top-hat initial condition and periodic boundaries is used to assess the MUSCL/TVD schemes implemented in `CBm0.numerics`. The limiter behaviour (Minmod, Superbee, Van Leer) follows the expected TVD characteristics described by van Leer [@vanLeer1977] and Sweby [@Sweby1984]: for a fixed Courant number and advection time, the Minmod limiter produces the most diffusive profile with an interface spread over roughly six grid cells, while Superbee and Van Leer resolve the front within about two cells without overshoots. These results, including the monotone limiter comparison, are summarised in `advection_top_hat_limiters.png`.
- **0D ignition and reaction kinetics**: Spatially homogeneous, zero-dimensional reactor tests are provided in `examples/run_ignition_0d.py` to verify the consistency of the global reaction mechanisms and thermodynamic property evaluation. For the supplied methane/air test case, coarse and reference time steps produce nearly indistinguishable temperature histories; the post-processing script reports a relative $L^2$ error in $T(t)$ that is effectively zero within numerical precision, and ignition times are therefore indistinguishable for the chosen parameters. The corresponding comparison figure is written by `examples/postprocess_ignition_0d.py`.
- **Buoyant diffusion flame and thermal plume (full reacting-flow solver)**: A laminar, buoyancy-driven diffusion flame in a vertical channel, driven by a fuel jet into coflowing oxidiser, is provided as a representative reacting-flow case in `examples/run_demo.py`. The simulated centreline temperature decay and flame height show the qualitative trends expected from classic fire-plume experiments and correlations (e.g., Heskestad [@Heskestad1983]; McCaffrey [@McCaffrey1979]). This case exercises the full solver (advection, diffusion, projection, variable density, and heat release) and is also used to explore grid-sensitivity by running on multiple meshes and comparing the final temperature fields and centreline profiles.

# References














