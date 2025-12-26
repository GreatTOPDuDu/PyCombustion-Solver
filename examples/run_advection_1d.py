import os
import sys

import numpy as np
import yaml

# Add project root to path so we can import CBm0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CBm0.numerics import tvd_div


def initial_top_hat(x: np.ndarray, x0: float = 0.2, x1: float = 0.4) -> np.ndarray:
    phi = np.zeros_like(x)
    phi[(x >= x0) & (x <= x1)] = 1.0
    return phi


def run_advection_1d():
    """1D-like scalar advection test using CBm0 MUSCL/TVD schemes.

    A top-hat scalar pulse is advected to the right with constant
    velocity. The solution after one domain traversal is written for
    different TVD limiters, demonstrating monotonicity and numerical
    diffusion characteristics.
    """

    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'advection_config.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Require these keys in advection_config.yaml so that the setup
    # is fully specified by the YAML file.
    Nx = int(cfg['Nx'])
    Ny = int(cfg['Ny'])
    L = float(cfg['L'])
    U = float(cfg['U'])
    cfl_adv = float(cfg['cfl_adv'])
    t_final = float(cfg['T_final'])
    limiters = list(cfg.get('limiters', ['minmod', 'superbee', 'vanleer']))

    dx = L / (Nx - 1)
    dy = 1.0  # thin domain in y; value does not matter much here

    x = np.linspace(0.0, L, Nx)

    out_dir = os.path.join(here, 'advection_output')
    os.makedirs(out_dir, exist_ok=True)

    for limiter in limiters:
        # Initialize scalar pulse, extend uniformly in y
        phi_line = initial_top_hat(x)
        phi = np.tile(phi_line[np.newaxis, :], (Ny, 1))

        # Uniform velocity field in x, zero in y
        u = U * np.ones_like(phi)
        v = np.zeros_like(phi)

        t = 0.0
        step = 0

        while t < t_final:
            umax = max(1e-8, float(np.max(np.abs(u))))
            vmax = max(1e-8, float(np.max(np.abs(v))))
            dt = cfl_adv / (umax / dx + vmax / dy + 1e-12)
            if t + dt > t_final:
                dt = t_final - t
            if dt <= 0.0:
                break

            phi0 = phi.copy()
            conv_phi = tvd_div(phi0, u, v, dx, dy, limiter_kind=limiter)
            phi = phi0 - dt * conv_phi

            # Periodic boundary conditions in x so that the top-hat
            # pulse remains in the domain after advection.
            phi[:, 0] = phi[:, -2]
            phi[:, -1] = phi[:, 1]

            t += dt
            step += 1

        # Take the middle row as 1D result
        mid_j = Ny // 2
        phi_1d = phi[mid_j, :]

        out_path = os.path.join(out_dir, f'top_hat_{limiter}.csv')
        np.savetxt(
            out_path,
            np.column_stack((x, phi_1d)),
            header=f'x,phi(x) after t={t_final} with limiter={limiter}',
            delimiter=',',
        )

        print(f"Advection test with limiter={limiter} finished in {step} steps. Output: {out_path}")


if __name__ == '__main__':
    run_advection_1d()
