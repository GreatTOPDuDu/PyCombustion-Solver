import os
import sys

import numpy as np
import yaml

# Add project root to path so we can import CBm0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CBm0.physics import Rmix_and_rho, cp_mixture_mass
from CBm0.chemistry import compute_sources_wd2


def run_ignition_0d():
    """Zero-dimensional ignition test using CBm0 chemistry.

    This script integrates a spatially homogeneous CH4/air mixture
    using the global Westbrook--Dryer mechanism. It writes the
    temperature and fuel mass fraction history to a CSV file and
    reports an approximate ignition delay time.
    """

    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, 'ignition_0d_config.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Single "cell" represented as 1x1 arrays
    shape = (1, 1)

    # Initial conditions from YAML
    T0 = float(cfg['T0'])
    YF0 = float(cfg['YF0'])
    YO0 = float(cfg['YO0'])
    P0 = float(cfg['P0'])
    fuel_type = str(cfg.get('fuel_type', 'CH4'))

    T = np.full(shape, T0)
    YF = np.full(shape, YF0)
    YO = np.full(shape, YO0)
    YC = np.zeros(shape)
    YC2 = np.zeros(shape)
    YW = np.zeros(shape)

    t_final = float(cfg['T_final'])    # final time [s]
    dt = float(cfg['dt'])              # time step [s]

    n_steps = int(round(t_final / dt))

    times = np.zeros(n_steps + 1)
    T_hist = np.zeros(n_steps + 1)
    YF_hist = np.zeros(n_steps + 1)

    # Initial diagnostics
    Rmix, rho = Rmix_and_rho(T, YF, YO, YC, YC2, YW, P0, fuel_type=fuel_type)
    Cp_mix = cp_mixture_mass(T, YF, YO, YC, YC2, YW, fuel_type=fuel_type)

    times[0] = 0.0
    T_hist[0] = float(T[0, 0])
    YF_hist[0] = float(YF[0, 0])

    t = 0.0
    t_ign = None

    for k in range(1, n_steps + 1):
        # Update thermo
        Rmix, rho = Rmix_and_rho(T, YF, YO, YC, YC2, YW, P0, fuel_type=fuel_type)
        Cp_mix = cp_mixture_mass(T, YF, YO, YC, YC2, YW, fuel_type=fuel_type)

        # Reaction source terms (Westbrook--Dryer 2-step)
        S_T, S_YF, S_YO, S_YC, S_YC2, S_YW, HRR_chem = compute_sources_wd2(
            T, rho, Cp_mix, YF, YO, YC, YC2, YW,
            A1_mult=1.0, Ea1_mult=1.0, A2_mult=1.0, Ea2_mult=1.0,
        )

        # Explicit Euler update
        T = T + dt * S_T
        YF = YF + dt * S_YF
        YO = YO + dt * S_YO
        YC = YC + dt * S_YC
        YC2 = YC2 + dt * S_YC2
        YW = YW + dt * S_YW

        # Basic clipping to keep variables physical
        T = np.clip(T, 200.0, 5000.0)
        for Y in (YF, YO, YC, YC2, YW):
            np.clip(Y, 0.0, 1.0, out=Y)
        Ysum = YF + YO + YC + YC2 + YW
        scale = np.where(Ysum > 1e-12, np.minimum(1.0, 1.0 / Ysum), 1.0)
        YF *= scale
        YO *= scale
        YC *= scale
        YC2 *= scale
        YW *= scale

        t += dt
        times[k] = t
        T_hist[k] = float(T[0, 0])
        YF_hist[k] = float(YF[0, 0])

        # Simple ignition criterion: temperature rise of 400 K
        if t_ign is None and T_hist[k] - T_hist[0] > 400.0:
            t_ign = t

    out_dir = os.path.join(os.path.dirname(__file__), 'ignition_output')
    os.makedirs(out_dir, exist_ok=True)

    data = np.column_stack((times, T_hist, YF_hist))
    out_path = os.path.join(out_dir, 'ignition_0d_history.csv')
    np.savetxt(
        out_path,
        data,
        header='t [s], T [K], YF',
        delimiter=',',
    )

    print(f"0D ignition run finished at t={t:.4e} s.")
    if t_ign is not None:
        print(f"Approximate ignition delay time: t_ign = {t_ign:.4e} s")
    else:
        print("No ignition detected with the current settings.")
    print(f"History written to {out_path}.")


if __name__ == '__main__':
    run_ignition_0d()
