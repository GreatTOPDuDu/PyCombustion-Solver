from __future__ import annotations
import os
import sys
import time
from typing import Tuple

import numpy as np
import numba
from tqdm import tqdm

from .config import Config
from .physics import (
    R_u, M_O2, M_N2,
    cp_mixture_mass, mu_species_all, mole_fractions_all, wilke_mixture_viscosity_general,
    Rmix_and_rho, k_mixture, D_species_T_all,
)
from .numerics import tvd_div, apply_diffusion, divergence, mg_solve
from .chemistry import compute_sources_wd2, compute_sources_h2_global, thermal_NO_source
from .io_utils import (
    get_output_dirs,
    save_contour,
    save_contour_log,
    write_run_config,
    get_linear_range_from_config,
    get_log_range_from_config,
)


def _convective_outflow_update(phi: np.ndarray, c_out: float, dt: float, dy: float) -> np.ndarray:
    if phi.shape[0] >= 3 and c_out > 0.0:
        phi[-1, :] = phi[-2, :] - c_out * dt / dy * (phi[-2, :] - phi[-3, :])
    else:
        phi[-1, :] = phi[-2, :]
    return phi


def main(cfg: Config) -> int:
    # Numba threads control
    if isinstance(cfg.num_threads, int) and cfg.num_threads > 0:
        try:
            numba.set_num_threads(cfg.num_threads)
        except Exception:
            pass

    P0 = cfg.P0_pa  # bind pressure

    # Paths
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    out_dirs = get_output_dirs(cfg, current_dir)

    # write configuration snapshot (include parallel info)
    extra_info = {}
    try:
        extra_info['numba_num_threads'] = int(numba.get_num_threads())
    except Exception:
        extra_info['numba_num_threads'] = 'unknown'
    extra_info['parallel_functions'] = ['numerics.smooth_rbgs (njit, prange)', 'numerics.restrict_avg (njit, prange)']
    write_run_config(cfg, out_dirs['method'], extra=extra_info)

    # Grid
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    dx = Lx / (Nx - 1); dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx); y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Inlet masks (fuel spans or uniform channels)
    fuel_mask = np.zeros(Nx, dtype=bool)
    if getattr(cfg, 'inlet_mode', 'uniform') == 'explicit' and len(cfg.fuel_inlet_spans_m) > 0:
        for (x0_m, x1_m) in cfg.fuel_inlet_spans_m:
            i0 = int(round(x0_m / dx)); i1 = int(round(x1_m / dx))
            i0 = max(0, min(Nx - 1, i0)); i1 = max(0, min(Nx - 1, i1))
            if i1 < i0:
                i0, i1 = i1, i0
            if i1 == i0:
                i1 = min(Nx - 1, i0 + 1)
            fuel_mask[i0:i1 + 1] = True
    else:
        channel_width_cells = max(1, int(round(cfg.fuel_channel_width / dx)))
        empty_space = Nx - cfg.num_fuel_channels * channel_width_cells
        gap = max(1, empty_space // (cfg.num_fuel_channels + 1))
        cursor = 0
        for _ in range(cfg.num_fuel_channels):
            start = cursor + gap; end = min(Nx, start + channel_width_cells)
            fuel_mask[start:end] = True; cursor = end

    # Build inlet profiles
    T_inlet = np.full(Nx, cfg.T_air)
    YF_inlet = np.full(Nx, cfg.YF_air)
    YO_inlet = np.full(Nx, cfg.YO_air)
    v_inlet = np.where(fuel_mask, cfg.v_fuel, cfg.v_air)

    if cfg.mixing_mode == 'premixed':
        phi_map = {'rich': 1.2, 'stoic': 1.0, 'lean': 0.8}
        phi_use = cfg.phi_override if (cfg.phi_override is not None and cfg.phi_override > 1e-6) else phi_map.get(cfg.equiv_mode, 1.0)
        # NOTE: r_st is for CH4. If using H2, this value/configuration needs separate adjustment.
        r_st = 4.0
        YO_air_mass = cfg.YO_air if cfg.YO_air > 1e-9 else 0.233
        m_O2_per_mF = r_st / max(phi_use, 1e-6)
        m_air = m_O2_per_mF / YO_air_mass
        m_total = 1.0 + m_air
        YF_mix = 1.0 / m_total
        YO_mix = m_O2_per_mF / m_total
        T_inlet[:] = cfg.T_air; YF_inlet[:] = YF_mix; YO_inlet[:] = YO_mix
        v_inlet = np.full(Nx, cfg.v_air)
    else:
        T_inlet[fuel_mask] = cfg.T_fuel
        YF_inlet[fuel_mask] = cfg.YF_fuel
        YO_inlet[fuel_mask] = cfg.YO_fuel

    # State arrays
    u = np.zeros((Ny, Nx)); v = np.zeros((Ny, Nx))
    T = np.full((Ny, Nx), cfg.T_air)
    YF = np.full((Ny, Nx), cfg.YF_air)
    YO = np.full((Ny, Nx), cfg.YO_air)
    YC = np.zeros((Ny, Nx)); YC2 = np.zeros((Ny, Nx)); YW = np.zeros((Ny, Nx))
    YNO = np.zeros((Ny, Nx))
    Pp = np.zeros((Ny, Nx))
    HRR_total = np.zeros((Ny, Nx))
    HRR_chem = np.zeros((Ny, Nx))

    ign_j = max(1, int(0.08 * Ny))

    # Initial saves
    Rmix, rho = Rmix_and_rho(T, YF, YO, YC, YC2, YW, P0, fuel_type=cfg.fuel_type)
    # Initial saves with config-driven ranges
    T_min, T_max = get_linear_range_from_config(cfg, 'T')
    YF_min, YF_max = get_linear_range_from_config(cfg, 'YF')
    YO_min, YO_max = get_linear_range_from_config(cfg, 'YO')
    YC_log_min, YC_log_max = get_log_range_from_config(cfg, 'YC')
    YC2_min, YC2_max = get_linear_range_from_config(cfg, 'YC2')
    YW_min, YW_max = get_linear_range_from_config(cfg, 'YW')
    RHO_min, RHO_max = get_linear_range_from_config(cfg, 'RHO')
    U_min, U_max = get_linear_range_from_config(cfg, 'U')
    V_min, V_max = get_linear_range_from_config(cfg, 'V')
    YNO_log_min, YNO_log_max = get_log_range_from_config(cfg, 'YNO')

    save_contour(T, X, Y, 0.0, 0, 'Temperature [K]', out_dirs['T'], T_min, T_max, 'inferno')
    save_contour(YF, X, Y, 0.0, 0, 'Fuel mass fraction', out_dirs['YF'], YF_min, YF_max, 'viridis')
    save_contour(YO, X, Y, 0.0, 0, 'O2 mass fraction', out_dirs['YO'], YO_min, YO_max, 'plasma')

    # Save initial CO/CO2 only in CH4-WD2 mode
    if cfg.kinetics_model == 'wd2' and 'YC' in out_dirs and 'YC2' in out_dirs:
        save_contour_log(
            YC, X, Y, 0.0, 0,
            'CO mass fraction',
            out_dirs['YC'],
            vmin_log=(YC_log_min if YC_log_min is not None else 1e-12),
            vmax=YC_log_max,
            cmap='magma',
        )
        save_contour(
            YC2, X, Y, 0.0, 0,
            'CO2 mass fraction',
            out_dirs['YC2'],
            YC2_min, YC2_max, 'cividis',
        )

    save_contour(YW, X, Y, 0.0, 0, 'H2O mass fraction', out_dirs['YW'], YW_min, YW_max, 'GnBu')
    save_contour(rho, X, Y, 0.0, 0, 'Density [kg/m^3]', out_dirs['RHO'], RHO_min, RHO_max, cmap='cividis')
    save_contour(u, X, Y, 0.0, 0, 'u velocity [m/s]', out_dirs['U'], U_min, U_max, cmap='coolwarm')
    save_contour(v, X, Y, 0.0, 0, 'v velocity [m/s]', out_dirs['V'], V_min, V_max, cmap='coolwarm')
    save_contour_log(
        YNO, X, Y, 0.0, 0,
        'NO mass fraction',
        out_dirs['YNO'],
        vmin_log=(YNO_log_min if YNO_log_min is not None else 1e-12),
        vmax=YNO_log_max,
        cmap='viridis',
    )

    # Diagnostics CSV
    os.makedirs(out_dirs['LOG'], exist_ok=True)
    diag_path = os.path.join(out_dirs['LOG'], 'diagnostics.csv')
    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write('step,time,dt,HRR_total_int(W/m),HRR_chem_int(W/m),E_chem_cum(J/m),mF(kg/m),d_mF(kg/m),Tmax,Tmean,RHOmin,RHOmax,RHOmean,Ysum_min,Ysum_max\n')

    def fuel_mass_integral(rho_f, YF_fuel):
        return float(np.sum(rho_f * YF_fuel) * dx * dy)

    mF0 = fuel_mass_integral(rho, YF)

    # Loop
    t = 0.0; step = 0
    pbar = tqdm(total=cfg.t_final, desc='CBm0 (MUSCL/TVD + WD2/H2, parallel)', unit='s')
    last_tick = time.time(); last_save_time = 0.0

    t_ign: float | None = None
    E_chem_cum = 0.0

    while t < cfg.t_final:
        Cp_mix = cp_mixture_mass(T, YF, YO, YC, YC2, YW, fuel_type=cfg.fuel_type)
        mu_map = mu_species_all(T, fuel_type=cfg.fuel_type)
        x_CH4, x_O2, x_CO, x_CO2, x_H2O, x_N2 = mole_fractions_all(YF, YO, YC, YC2, YW, fuel_type=cfg.fuel_type)
        # First component corresponds to "fuel" (CH4 or H2), key is fixed to CH4
        if cfg.fuel_type.upper() == "H2":
            M_fuel = 2e-3
        else:
            M_fuel = 16e-3
        mu_mix = wilke_mixture_viscosity_general(
            [mu_map['CH4'], mu_map['O2'], mu_map['CO'], mu_map['CO2'], mu_map['H2O'], mu_map['N2']],
            [x_CH4, x_O2, x_CO, x_CO2, x_H2O, x_N2],
            [M_fuel, 32e-3, 28e-3, 44e-3, 18e-3, 28e-3],
        )
        k_mix = k_mixture(mu_mix, Cp_mix, Pr=cfg.Pr_mix_ref)
        D_ch4_or_h2, D_o2, D_co, D_co2, D_h2o, D_n2 = D_species_T_all(T, fuel_type=cfg.fuel_type)
        Rmix_loc, rho = Rmix_and_rho(T, YF, YO, YC, YC2, YW, P0, fuel_type=cfg.fuel_type)
        nu_loc = mu_mix / np.maximum(rho, 1e-12)
        alpha_loc = k_mix / (rho * np.maximum(Cp_mix, 1e-12))
        Dmax = np.maximum.reduce([D_ch4_or_h2, D_o2, D_co, D_co2, D_h2o])
        umax = max(1e-8, float(np.max(np.abs(u))))
        vmax = max(1e-8, float(np.max(np.abs(v))))
        dt_adv = cfg.cfl_adv / (umax / dx + vmax / dy + 1e-12)
        dt_diff = cfg.cfl_diff / (max(float(np.max(alpha_loc)), float(np.max(nu_loc)), float(np.max(Dmax))) *
                                  (1.0 / dx**2 + 1.0 / dy**2) + 1e-12)
        dt = min(dt_adv, dt_diff)
        if t + dt > cfg.t_final:
            dt = cfg.t_final - t
        if dt <= 1e-12:
            break

        u0, v0 = u.copy(), v.copy()
        T0, YF0, YO0, YC0, YC20, YW0, YNO0 = (
            T.copy(), YF.copy(), YO.copy(), YC.copy(), YC2.copy(), YW.copy(), YNO.copy()
        )

        # Transport
        conv_T = tvd_div(T0, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)
        conv_YF = tvd_div(YF0, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)
        conv_YO = tvd_div(YO0, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)
        conv_YC = tvd_div(YC0, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)
        conv_YC2 = tvd_div(YC20, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)
        conv_YW = tvd_div(YW0, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)
        conv_YNO = tvd_div(YNO0, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)

        diff_T = apply_diffusion(T0, alpha_loc, dx, dy)
        diff_YF = apply_diffusion(YF0, D_ch4_or_h2, dx, dy)
        diff_YO = apply_diffusion(YO0, D_o2, dx, dy)
        diff_YC = apply_diffusion(YC0, D_co, dx, dy)
        diff_YC2 = apply_diffusion(YC20, D_co2, dx, dy)
        diff_YW = apply_diffusion(YW0, D_h2o, dx, dy)
        diff_YNO = apply_diffusion(YNO0, D_n2, dx, dy)  # Assume NO ~ N2 diffusion coefficient

        T = T0 + dt * (diff_T - conv_T)
        YF = YF0 + dt * (diff_YF - conv_YF)
        YO = YO0 + dt * (diff_YO - conv_YO)
        YC = YC0 + dt * (diff_YC - conv_YC)
        YC2 = YC20 + dt * (diff_YC2 - conv_YC2)
        YW = YW0 + dt * (diff_YW - conv_YW)
        YNO = YNO0 + dt * (diff_YNO - conv_YNO)

        # Scalar BCs
        T[0, :] = T_inlet; YF[0, :] = YF_inlet; YO[0, :] = YO_inlet
        YC[0, :] = 0.0; YC2[0, :] = 0.0; YW[0, :] = 0.0; YNO[0, :] = 0.0

        T[:, 0] = T[:, 1];   T[:, -1] = T[:, -2]
        YF[:, 0] = YF[:, 1]; YF[:, -1] = YF[:, -2]
        YO[:, 0] = YO[:, 1]; YO[:, -1] = YO[:, -2]
        YC[:, 0] = YC[:, 1]; YC[:, -1] = YC[:, -2]
        YC2[:, 0] = YC2[:, 1]; YC2[:, -1] = YC2[:, -2]
        YW[:, 0] = YW[:, 1]; YW[:, -1] = YW[:, -2]
        YNO[:, 0] = YNO[:, 1]; YNO[:, -1] = YNO[:, -2]

        # Chemistry subcycling
        n_sub = max(1, cfg.CHEM_SUBSTEPS); dt_sub = dt / n_sub
        E_spark_this_step = 0.0
        for sub_k in range(n_sub):
            Cp_mix_sub = cp_mixture_mass(T, YF, YO, YC, YC2, YW, fuel_type=cfg.fuel_type)
            _, rho_sub = Rmix_and_rho(T, YF, YO, YC, YC2, YW, P0, fuel_type=cfg.fuel_type)

            if cfg.kinetics_model == 'wd2':
                S_T, S_YF, S_YO, S_YC, S_YC2, S_YW, HRR_chem_sub = compute_sources_wd2(
                    T, rho_sub, Cp_mix_sub, YF, YO, YC, YC2, YW
                )
            elif cfg.kinetics_model == 'h2':
                S_T, S_YF, S_YO, S_YC, S_YC2, S_YW, HRR_chem_sub = compute_sources_h2_global(
                    T, rho_sub, Cp_mix_sub, YF, YO, YW
                )
            else:
                S_T = np.zeros_like(T); S_YF = np.zeros_like(T); S_YO = np.zeros_like(T)
                S_YC = np.zeros_like(T); S_YC2 = np.zeros_like(T); S_YW = np.zeros_like(T)
                HRR_chem_sub = np.zeros_like(T)

            # Thermal NOx generation
            if cfg.enable_thermal_NOx:
                S_YNO = thermal_NO_source(T, rho_sub, YF, YO, YC, YC2, YW)
            else:
                S_YNO = np.zeros_like(T)

            HRR_total_sub = HRR_chem_sub.copy()

            # substep update
            T = T + dt_sub * S_T
            YF = YF + dt_sub * S_YF
            YO = YO + dt_sub * S_YO
            YC = YC + dt_sub * S_YC
            YC2 = YC2 + dt_sub * S_YC2
            YW = YW + dt_sub * S_YW
            YNO = YNO + dt_sub * S_YNO

            HRR_total = HRR_total_sub
            HRR_chem = HRR_chem_sub

            E_chem_cum += float(np.sum(HRR_chem_sub) * dt_sub * dx * dy)

            # stabilization
            T = np.clip(T, 1.0, 4500.0)
            for Yvar in (YF, YO, YC, YC2, YW, YNO):
                np.clip(Yvar, 0.0, 1.0, out=Yvar)
            Ysum = YF + YO + YC + YC2 + YW + YNO
            scale = np.where(Ysum > 1e-12, np.minimum(1.0, 1.0 / Ysum), 1.0)
            YF *= scale; YO *= scale; YC *= scale; YC2 *= scale; YW *= scale; YNO *= scale

        # Outflow update
        c_out = max(0.0, float(np.mean(v0[-2, :])))
        T = _convective_outflow_update(T, c_out, dt, dy)
        YF = _convective_outflow_update(YF, c_out, dt, dy)
        YO = _convective_outflow_update(YO, c_out, dt, dy)
        YC = _convective_outflow_update(YC, c_out, dt, dy)
        YC2 = _convective_outflow_update(YC2, c_out, dt, dy)
        YW = _convective_outflow_update(YW, c_out, dt, dy)
        YNO = _convective_outflow_update(YNO, c_out, dt, dy)

        # Renormalize
        T = np.clip(T, 1.0, 4500.0)
        for Yvar in (YF, YO, YC, YC2, YW, YNO):
            np.clip(Yvar, 0.0, 1.0, out=Yvar)
        Ysum = YF + YO + YC + YC2 + YW + YNO
        scale = np.where(Ysum > 1e-12, np.minimum(1.0, 1.0 / Ysum), 1.0)
        YF *= scale; YO *= scale; YC *= scale; YC2 *= scale; YW *= scale; YNO *= scale

        # Momentum predictor
        mu_map = mu_species_all(T, fuel_type=cfg.fuel_type)
        x_CH4, x_O2, x_CO, x_CO2, x_H2O, x_N2 = mole_fractions_all(YF, YO, YC, YC2, YW, fuel_type=cfg.fuel_type)
        if cfg.fuel_type.upper() == "H2":
            M_fuel = 2e-3
        else:
            M_fuel = 16e-3
        mu_mix = wilke_mixture_viscosity_general(
            [mu_map['CH4'], mu_map['O2'], mu_map['CO'], mu_map['CO2'], mu_map['H2O'], mu_map['N2']],
            [x_CH4, x_O2, x_CO, x_CO2, x_H2O, x_N2],
            [M_fuel, 32e-3, 28e-3, 44e-3, 18e-3, 28e-3],
        )
        nu_loc = mu_mix / np.maximum(rho, 1e-12)
        conv_u = tvd_div(u0, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)
        conv_v = tvd_div(v0, u0, v0, dx, dy, limiter_kind=cfg.adv_limiter)
        diff_u = apply_diffusion(u0, nu_loc, dx, dy)
        diff_v = apply_diffusion(v0, nu_loc, dx, dy)
        denom_air = cfg.YO_air / M_O2 + (1.0 - cfg.YO_air) / M_N2
        R_air = R_u / (1.0 / denom_air)
        rho_ref = P0 / (R_air * cfg.T_air)
        buoy = 9.81 * (rho_ref - rho) / max(rho_ref, 1e-12)
        u_star = u0 + dt * (diff_u - conv_u)
        v_star = v0 + dt * (diff_v - conv_v + buoy)

        # Velocity BCs
        u_star[0, :] = 0.0; v_star[0, :] = v_inlet
        u_star[:, 0] = 0.0; u_star[:, -1] = 0.0
        v_star[:, 0] = 0.0; v_star[:, -1] = 0.0
        u_star = _convective_outflow_update(u_star, c_out, dt, dy)
        v_star = _convective_outflow_update(v_star, c_out, dt, dy)

        # Pressure projection
        kcoef = 1.0 / np.maximum(rho, 1e-12)
        rhs = (1.0 / dt) * divergence(u_star, v_star, dx, dy)
        for _ in range(cfg.mg_cycles_per_step):
            Pp = mg_solve(
                rhs, kcoef, dx, dy, Pp, cycles=1,
                pre=cfg.mg_pre_smooth, post=cfg.mg_post_smooth,
                mgmin=cfg.mg_coarsest_min,
            )
        dpdx = np.zeros_like(Pp); dpdy = np.zeros_like(Pp)
        if Pp.shape[1] >= 3:
            dpdx[:, 1:-1] = (Pp[:, 2:] - Pp[:, :-2]) / (2.0 * dx)
        if Pp.shape[0] >= 3:
            dpdy[1:-1, :] = (Pp[2:, :] - Pp[:-2, :]) / (2.0 * dy)
        dpdx[:, 0] = (Pp[:, 1] - Pp[:, 0]) / dx; dpdx[:, -1] = (Pp[:, -1] - Pp[:, -2]) / dx
        dpdy[0, :] = (Pp[1, :] - Pp[0, :]) / dy; dpdy[-1, :] = (Pp[-1, :] - Pp[-2, :]) / dy
        u = u_star - dt * kcoef * dpdx; v = v_star - dt * kcoef * dpdy
        u[0, :] = 0.0; v[0, :] = v_inlet
        u[:, 0] = 0.0; u[:, -1] = 0.0; v[:, 0] = 0.0; v[:, -1] = 0.0
        u = _convective_outflow_update(u, c_out, dt, dy)
        v = _convective_outflow_update(v, c_out, dt, dy)
        Pp[-1, :] = 0.0

        # Time update
        t += dt; step += 1
        if time.time() - last_tick > 0.1:
            pbar.n = t; pbar.refresh(); last_tick = time.time()

        # Diagnostics
        HRR_total_int = float(np.sum(HRR_total) * dx * dy)
        HRR_chem_int = float(np.sum(HRR_chem) * dx * dy)
        Tmax = float(np.max(T)); Tmean = float(np.mean(T))
        mF = fuel_mass_integral(rho, YF); d_mF = mF0 - mF
        rho_min = float(np.min(rho)); rho_max = float(np.max(rho)); rho_mean = float(np.mean(rho))
        Ysum_diag = YF + YO + YC + YC2 + YW + YNO
        with open(diag_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{step},{t:.6e},{dt:.6e},{HRR_total_int:.6e},{HRR_chem_int:.6e},"
                f"{E_chem_cum:.6e},{mF:.6e},{d_mF:.6e},{Tmax:.3f},{Tmean:.3f},"
                f"{rho_min:.6e},{rho_max:.6e},{rho_mean:.6e},"
                f"{float(np.min(Ysum_diag)):.6e},{float(np.max(Ysum_diag)):.6e}\n"
            )

        # Ignition detection
        if (t_ign is None) and (HRR_chem_int > cfg.ign_hrr_threshold_Wm) and (Tmax > 1700.0):
            t_ign = t
            with open(os.path.join(out_dirs['LOG'], 'ignition.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Ignition detected at t={t_ign:.6e} s\n")

        # Saves (with log-scale for HRR and CO fraction)
        save_step_trigger = (cfg.save_every is not None and step % cfg.save_every == 0)
        save_time_trigger = (cfg.save_interval is not None and (t - last_save_time >= cfg.save_interval))
        if save_step_trigger or save_time_trigger or (t >= cfg.t_final):
            T_min, T_max = get_linear_range_from_config(cfg, 'T')
            YF_min, YF_max = get_linear_range_from_config(cfg, 'YF')
            YO_min, YO_max = get_linear_range_from_config(cfg, 'YO')
            YC_log_min, YC_log_max = get_log_range_from_config(cfg, 'YC')
            YC2_min, YC2_max = get_linear_range_from_config(cfg, 'YC2')
            YW_min, YW_max = get_linear_range_from_config(cfg, 'YW')
            HRR_min, HRR_max = get_log_range_from_config(cfg, 'HRR')
            RHO_min, RHO_max = get_linear_range_from_config(cfg, 'RHO')
            U_min, U_max = get_linear_range_from_config(cfg, 'U')
            V_min, V_max = get_linear_range_from_config(cfg, 'V')
            P_min, P_max = get_linear_range_from_config(cfg, 'P')
            YNO_log_min, YNO_log_max = get_log_range_from_config(cfg, 'YNO')

            save_contour(T, X, Y, t, step, 'Temperature [K]', out_dirs['T'], T_min, T_max, 'inferno')
            save_contour(YF, X, Y, t, step, 'Fuel mass fraction', out_dirs['YF'], YF_min, YF_max, 'viridis')
            save_contour(YO, X, Y, t, step, 'O2 mass fraction', out_dirs['YO'], YO_min, YO_max, 'plasma')

            # Save CO/CO2 only in CH4-WD2 mode
            if cfg.kinetics_model == 'wd2' and 'YC' in out_dirs and 'YC2' in out_dirs:
                save_contour_log(
                    YC, X, Y, t, step,
                    'CO mass fraction',
                    out_dirs['YC'],
                    vmin_log=(YC_log_min if YC_log_min is not None else 1e-10),
                    vmax=YC_log_max,
                    cmap='magma',
                )
                save_contour(
                    YC2, X, Y, t, step,
                    'CO2 mass fraction',
                    out_dirs['YC2'],
                    YC2_min, YC2_max,
                    'cividis',
                )

            save_contour(YW, X, Y, t, step, 'H2O mass fraction', out_dirs['YW'], YW_min, YW_max, 'GnBu')
            save_contour_log(
                HRR_total, X, Y, t, step,
                'Heat Release Rate total [W/m^3]', out_dirs['HRR'],
                vmin_log=(HRR_min if HRR_min is not None else 1e-1),
                vmax=HRR_max,
                cmap='magma',
            )
            _Rmix_s, rho_save = Rmix_and_rho(T, YF, YO, YC, YC2, YW, P0, fuel_type=cfg.fuel_type)
            save_contour(rho_save, X, Y, t, step, 'Density [kg/m^3]', out_dirs['RHO'], RHO_min, RHO_max, cmap='cividis')
            save_contour(Pp, X, Y, t, step, 'Projection pressure [Pa]', out_dirs['P'], P_min, P_max, cmap='magma')
            save_contour(u, X, Y, t, step, 'u velocity [m/s]', out_dirs['U'], U_min, U_max, cmap='coolwarm')
            save_contour(v, X, Y, t, step, 'v velocity [m/s]', out_dirs['V'], V_min, V_max, cmap='coolwarm')
            save_contour_log(
                YNO, X, Y, t, step,
                'NO mass fraction',
                out_dirs['YNO'],
                vmin_log=(YNO_log_min if YNO_log_min is not None else 1e-12),
                vmax=YNO_log_max,
                cmap='viridis',
            )
            last_save_time = t

    pbar.close()

    # Final saves
    T_min, T_max = get_linear_range_from_config(cfg, 'T')
    YF_min, YF_max = get_linear_range_from_config(cfg, 'YF')
    YO_min, YO_max = get_linear_range_from_config(cfg, 'YO')
    YC_log_min, YC_log_max = get_log_range_from_config(cfg, 'YC')
    YC2_min, YC2_max = get_linear_range_from_config(cfg, 'YC2')
    YW_min, YW_max = get_linear_range_from_config(cfg, 'YW')
    YNO_log_min, YNO_log_max = get_log_range_from_config(cfg, 'YNO')

    save_contour(T, X, Y, t, step, 'Temperature [K]', out_dirs['T'], T_min, T_max, 'inferno')
    save_contour(YF, X, Y, t, step, 'Fuel mass fraction', out_dirs['YF'], YF_min, YF_max, 'viridis')
    save_contour(YO, X, Y, t, step, 'O2 mass fraction', out_dirs['YO'], YO_min, YO_max, 'plasma')

    # Save final CO/CO2 only in CH4-WD2 mode
    if cfg.kinetics_model == 'wd2' and 'YC' in out_dirs and 'YC2' in out_dirs:
        save_contour_log(
            YC, X, Y, t, step,
            'CO mass fraction',
            out_dirs['YC'],
            vmin_log=(YC_log_min if YC_log_min is not None else 1e-10),
            vmax=YC_log_max,
            cmap='magma',
        )
        save_contour(
            YC2, X, Y, t, step,
            'CO2 mass fraction',
            out_dirs['YC2'],
            YC2_min, YC2_max,
            'cividis',
        )

    save_contour(YW, X, Y, t, step, 'H2O mass fraction', out_dirs['YW'], YW_min, YW_max, 'GnBu')
    _Rmix_s, rho_final = Rmix_and_rho(T, YF, YO, YC, YC2, YW, P0, fuel_type=cfg.fuel_type)
    RHO_min, RHO_max = get_linear_range_from_config(cfg, 'RHO')
    HRR_min, HRR_max = get_log_range_from_config(cfg, 'HRR')
    U_min, U_max = get_linear_range_from_config(cfg, 'U')
    V_min, V_max = get_linear_range_from_config(cfg, 'V')
    save_contour(rho_final, X, Y, t, step, 'Density [kg/m^3]', out_dirs['RHO'], RHO_min, RHO_max, cmap='cividis')
    save_contour_log(
        HRR_total, X, Y, t, step,
        'Heat Release Rate total [W/m^3]', out_dirs['HRR'],
        vmin_log=(HRR_min if HRR_min is not None else 1e-1),
        vmax=HRR_max,
        cmap='magma',
    )
    save_contour(u, X, Y, t, step, 'u velocity [m/s]', out_dirs['U'], U_min, U_max, cmap='coolwarm')
    save_contour(v, X, Y, t, step, 'v velocity [m/s]', out_dirs['V'], V_min, V_max, cmap='coolwarm')
    save_contour_log(
        YNO, X, Y, t, step,
        'NO mass fraction',
        out_dirs['YNO'],
        vmin_log=(YNO_log_min if YNO_log_min is not None else 1e-12),
        vmax=YNO_log_max,
        cmap='viridis',
    )

    print(f"\nCBm0 finished at t={t:.4f} s. P0={P0:.0f} Pa. Results in {out_dirs['method']}")
    ign_path = os.path.join(out_dirs['LOG'], 'ignition.txt')
    if os.path.exists(ign_path):
        with open(ign_path, 'r', encoding='utf-8') as f:
            print(f.read().strip())
    return 0


if __name__ == '__main__':
    from .config import parse_config
    print("Numba parallel JIT enabled. First run may be slower (compiling...)")
    cfg = parse_config()
    sys.exit(main(cfg))
