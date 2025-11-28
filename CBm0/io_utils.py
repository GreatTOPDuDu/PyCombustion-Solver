from __future__ import annotations
import os
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .config import Config


def get_output_dirs(cfg: Config, current_dir: str):
    # If output_base_dir is absolute, use it directly.
    # Otherwise, join with current_dir (cwd).
    if os.path.isabs(cfg.output_base_dir):
        output_dir_base = cfg.output_base_dir
    else:
        output_dir_base = os.path.join(current_dir, cfg.output_base_dir)

    output_dir_method = os.path.join(output_dir_base, cfg.output_method_dir)

    # Basic fields (CO, CO2 added only in wd2 mode)
    d: dict[str, str] = {
        'base': output_dir_base,
        'method': output_dir_method,
        'T': os.path.join(output_dir_method, 'Temperature'),
        'YF': os.path.join(output_dir_method, 'FuelFraction'),
        'YO': os.path.join(output_dir_method, 'O2Fraction'),
        'YW': os.path.join(output_dir_method, 'H2OFraction'),
        'YNO': os.path.join(output_dir_method, 'NOFraction'),
        'HRR': os.path.join(output_dir_method, 'HeatReleaseRate'),
        'P': os.path.join(output_dir_method, 'Pressure'),
        'RHO': os.path.join(output_dir_method, 'Density'),
        'U': os.path.join(output_dir_method, 'u_velocity'),
        'V': os.path.join(output_dir_method, 'v_velocity'),
        'LOG': os.path.join(output_dir_method, 'diagnostics'),
    }

    # Create CO/CO2 directory only in CH4-WD2 mode
    if getattr(cfg, "kinetics_model", "wd2") == "wd2":
        d['YC'] = os.path.join(output_dir_method, 'COFraction')
        d['YC2'] = os.path.join(output_dir_method, 'CO2Fraction')

    os.makedirs(d['method'], exist_ok=True)
    for key, path in d.items():
        if key in ('base', 'method'):
            continue
        os.makedirs(path, exist_ok=True)
    return d


def save_contour(field, X, Y, time_s, step, var_name, out_dir, vmin=None, vmax=None, cmap='viridis'):
    plt.figure(figsize=(6, 8))
    if vmin is None or vmax is None:
        vmin = float(np.nanmin(field)); vmax = float(np.nanmax(field))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            vmin, vmax = 0.0, 1.0
    lv = np.linspace(vmin, vmax, 51)
    cp = plt.contourf(X, Y, field, levels=lv, cmap=cmap, extend='both')
    plt.colorbar(cp, label=var_name)
    plt.title(f'{var_name} t={time_s:.4f}s (#{step})')
    plt.xlabel('X [m]'); plt.ylabel('Y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'frame_{step:05d}.png'))
    plt.close()


def save_contour_log(field, X, Y, time_s, step, var_name, out_dir, vmin_log=1e-12, vmax=None, cmap='magma'):
    # Ensure strictly positive for LogNorm
    fld = np.asarray(field)
    if vmax is None:
        vmax = float(np.nanmax(fld)) if np.isfinite(np.nanmax(fld)) else vmin_log
    vmax = max(vmax, vmin_log * 10.0)
    fldp = np.clip(fld, vmin_log, vmax)

    plt.figure(figsize=(6, 8))
    cp = plt.contourf(X, Y, fldp, levels=51, cmap=cmap, norm=LogNorm(vmin=vmin_log, vmax=vmax))
    plt.colorbar(cp, label=var_name + ' (log scale)')
    plt.title(f'{var_name} t={time_s:.4f}s (#{step})')
    plt.xlabel('X [m]'); plt.ylabel('Y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'frame_{step:05d}.png'))
    plt.close()


# Helpers to fetch plotting ranges from config
def _sanitize_range(vmin: float | None, vmax: float | None):
    if vmin is None or vmax is None:
        return None, None
    try:
        vmin_f = float(vmin); vmax_f = float(vmax)
        if not np.isfinite(vmin_f) or not np.isfinite(vmax_f):
            return None, None
        if vmax_f <= vmin_f:
            return None, None
        return vmin_f, vmax_f
    except Exception:
        return None, None


def get_linear_range_from_config(cfg: Config, key: str) -> tuple[float | None, float | None]:
    key = key.upper()
    if key == 'T':
        return _sanitize_range(cfg.plot_T_min, cfg.plot_T_max)
    if key == 'YF':
        return _sanitize_range(cfg.plot_YF_min, cfg.plot_YF_max)
    if key == 'YO':
        return _sanitize_range(cfg.plot_YO_min, cfg.plot_YO_max)
    if key == 'YC2':
        return _sanitize_range(cfg.plot_YC2_min, cfg.plot_YC2_max)
    if key == 'YW':
        return _sanitize_range(cfg.plot_YW_min, cfg.plot_YW_max)
    if key == 'RHO':
        return _sanitize_range(cfg.plot_RHO_min, cfg.plot_RHO_max)
    if key == 'U':
        return _sanitize_range(cfg.plot_U_min, cfg.plot_U_max)
    if key == 'V':
        return _sanitize_range(cfg.plot_V_min, cfg.plot_V_max)
    if key == 'P':
        return _sanitize_range(cfg.plot_P_min, cfg.plot_P_max)
    return None, None


def get_log_range_from_config(cfg: Config, key: str) -> tuple[float | None, float | None]:
    key = key.upper()
    if key == 'YC':
        return _sanitize_range(cfg.plot_YC_min, cfg.plot_YC_max)
    if key == 'HRR':
        return _sanitize_range(cfg.plot_HRR_min, cfg.plot_HRR_max)
    if key == 'YNO':
        return _sanitize_range(cfg.plot_YNO_min, cfg.plot_YNO_max)
    return None, None


def write_run_config(cfg: Config, output_dir_method: str, extra: Optional[dict] = None) -> str:
    path = os.path.join(output_dir_method, 'run_config.txt')
    lines = []
    try:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    except Exception:
        timestamp = 'unknown'
    lines.append('CBm0 run configuration')
    lines.append(f'created_at={timestamp}')
    lines.append(f'output_base_dir={cfg.output_base_dir}')
    lines.append(f'output_method_dir={cfg.output_method_dir}')
    lines.append('')
    lines.append('# Config fields')
    try:
        cfg_dict = asdict(cfg)
    except Exception:
        cfg_dict = cfg.__dict__ if hasattr(cfg, '__dict__') else {}
    for k, v in cfg_dict.items():
        lines.append(f'{k}={v}')
    if extra:
        lines.append('')
        lines.append('# Derived/extra')
        for k, v in extra.items():
            lines.append(f'{k}={v}')
    os.makedirs(output_dir_method, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path




