import os
import sys

import numpy as np
import yaml

# Add project root to path so we can import CBm0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CBm0.numerics import tvd_div, apply_diffusion, divergence, mg_solve


def run_cavity():
    """Lid-driven cavity example using CBm0 numerics.

    This script solves the 2D incompressible lid-driven cavity with
    constant density and viscosity on a unit square. It exercises the
    advection, diffusion, and projection operators in CBm0.
    """

    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'cavity_config.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # These keys are required in cavity_config.yaml so that the
    # validation case is fully controlled by the config file.
    Nx = int(cfg['Nx'])
    Ny = int(cfg['Ny'])
    L = float(cfg['L'])
    Re = float(cfg['Re'])
    U_lid = float(cfg['U_lid'])

    cfl_adv = float(cfg['cfl_adv'])
    cfl_diff = float(cfg['cfl_diff'])
    t_final = float(cfg['T_final'])

    mg_cycles_per_step = int(cfg['mg_cycles_per_step'])
    mg_pre_smooth = int(cfg['mg_pre_smooth'])
    mg_post_smooth = int(cfg['mg_post_smooth'])
    mg_coarsest_min = int(cfg['mg_coarsest_min'])

    # Grid and parameters
    dx = L / (Nx - 1)
    dy = L / (Ny - 1)

    rho = np.ones((Ny, Nx), dtype=float)
    nu = U_lid * L / Re  # kinematic viscosity
    nu_field = nu * np.ones_like(rho)

    # Velocity and pressure fields
    u = np.zeros((Ny, Nx), dtype=float)
    v = np.zeros((Ny, Nx), dtype=float)
    p = np.zeros((Ny, Nx), dtype=float)

    t = 0.0
    step = 0

    out_dir = os.path.join(here, 'cavity_output')
    os.makedirs(out_dir, exist_ok=True)

    while t < t_final:
        umax = max(1e-8, float(np.max(np.abs(u))))
        vmax = max(1e-8, float(np.max(np.abs(v))))
        dt_adv = cfl_adv / (umax / dx + vmax / dy + 1e-12)
        dt_diff = cfl_diff / (nu * (1.0 / dx**2 + 1.0 / dy**2) + 1e-12)
        dt = min(dt_adv, dt_diff)
        if t + dt > t_final:
            dt = t_final - t
        if dt <= 0.0:
            break

        u0 = u.copy()
        v0 = v.copy()

        # Convective and diffusive terms
        conv_u = tvd_div(u0, u0, v0, dx, dy, limiter_kind='superbee')
        conv_v = tvd_div(v0, u0, v0, dx, dy, limiter_kind='superbee')
        diff_u = apply_diffusion(u0, nu_field, dx, dy)
        diff_v = apply_diffusion(v0, nu_field, dx, dy)

        # Momentum predictor
        u_star = u0 + dt * (diff_u - conv_u)
        v_star = v0 + dt * (diff_v - conv_v)

        # Lid-driven cavity boundary conditions (no penetration at walls)
        # Bottom wall (y=0)
        u_star[0, :] = 0.0
        v_star[0, :] = 0.0
        # Top wall (y=L): moving lid
        u_star[-1, :] = U_lid
        v_star[-1, :] = 0.0
        # Left/right walls
        u_star[:, 0] = 0.0
        u_star[:, -1] = 0.0
        v_star[:, 0] = 0.0
        v_star[:, -1] = 0.0

        # Projection step
        kcoef = 1.0 / rho
        rhs = (1.0 / dt) * divergence(u_star, v_star, dx, dy)
        for _ in range(mg_cycles_per_step):
            p = mg_solve(
                rhs,
                kcoef,
                dx,
                dy,
                p,
                cycles=1,
                pre=mg_pre_smooth,
                post=mg_post_smooth,
                mgmin=mg_coarsest_min,
            )

        dpdx = np.zeros_like(p)
        dpdy = np.zeros_like(p)
        if p.shape[1] >= 3:
            dpdx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dx)
        if p.shape[0] >= 3:
            dpdy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dy)
        dpdx[:, 0] = (p[:, 1] - p[:, 0]) / dx
        dpdx[:, -1] = (p[:, -1] - p[:, -2]) / dx
        dpdy[0, :] = (p[1, :] - p[0, :]) / dy
        dpdy[-1, :] = (p[-1, :] - p[-2, :]) / dy

        u = u_star - dt * kcoef * dpdx
        v = v_star - dt * kcoef * dpdy

        # Re-apply boundary conditions after projection
        u[0, :] = 0.0
        v[0, :] = 0.0
        u[-1, :] = U_lid
        v[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0

        t += dt
        step += 1

    # Save centerline velocity profiles for comparison with Ghia et al.
    i_mid = Nx // 2
    j_mid = Ny // 2
    u_centerline = u[:, i_mid]
    v_centerline = v[j_mid, :]

    np.savetxt(
        os.path.join(out_dir, 'u_centerline.csv'),
        np.column_stack((np.linspace(0.0, L, Ny), u_centerline)),
        header='y,u(y) at x=0.5',
        delimiter=',',
    )
    np.savetxt(
        os.path.join(out_dir, 'v_centerline.csv'),
        np.column_stack((np.linspace(0.0, L, Nx), v_centerline)),
        header='x,v(x) at y=0.5',
        delimiter=',',
    )

    print(f"Cavity run finished at t={t:.3f} (steps={step}).")
    print(f"Centerline data written to {out_dir}.")


if __name__ == '__main__':
    run_cavity()
