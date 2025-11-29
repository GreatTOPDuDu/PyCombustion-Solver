from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class Config:
    # grid
    Nx: int = 128
    Ny: int = 256
    Lx: float = 0.80
    Ly: float = 1.60

    # thermodynamic reference
    P0_pa: float = 1.0e5

    # inlets
    v_fuel: float = 0.8
    v_air: float = 0.5
    T_fuel: float = 380.0
    T_air: float = 1080.0
    YF_fuel: float = 1.0
    YO_fuel: float = 0.0
    YF_air: float = 0.0
    YO_air: float = 0.233

    # mixing mode
    mixing_mode: str = "stratified"  # 'stratified' | 'premixed'
    equiv_mode: str = "stoic"        # 'rich' | 'stoic' | 'lean'
    phi_override: Optional[float] = None
    num_fuel_channels: int = 1
    fuel_channel_width: float = 1.0e-3

    # inlet specification
    inlet_mode: str = 'uniform'  # 'uniform' | 'explicit'
    # explicit inlet spans: List of (y_start, y_end) in meters(set None to use uniform inlet)
    fuel_inlet_spans_m: List[Tuple[float, float]] = field(default_factory=lambda: [None])

    # time & numerics
    t_final: float = 3.0
    cfl_adv: float = 0.35
    cfl_diff: float = 0.35
    # Save condition: If None, that criterion is disabled
    save_every: Optional[int] = 50        # Step-based save interval (None -> disabled)
    save_interval: Optional[float] = None # Physical time interval save (None -> disabled)

    # pressure solver (multigrid)
    mg_pre_smooth: int = 2
    mg_post_smooth: int = 2
    mg_cycles_per_step: int = 2
    mg_coarsest_min: int = 8

    # transport
    Pr_mix_ref: float = 0.7
    adv_limiter: str = "superbee"  # 'minmod'|'vanleer'|'superbee'

    # chemistry
    kinetics_model: str = "wd2"   # 'wd2' | 'h2'
    CHEM_SUBSTEPS: int = 25
    fuel_type: str = "CH4"        # 'CH4' | 'H2'
    enable_thermal_NOx: bool = True

    # ignition
    ignition_mode: str = "off"     

    # ignition detection
    ign_hrr_threshold_Wm: float = 10.0

    # parallelization
    num_threads: int = 8  

    # output directories
    output_base_dir: str = 'out'
    output_method_dir: str = 'test1'

    # plotting ranges (set None to auto-scale). For log-scale vars (YC, HRR, YNO) use positive values.
    # Linear-scale plots
    plot_T_min: Optional[float] = 300.0
    plot_T_max: Optional[float] = 3500.0
    plot_YF_min: Optional[float] = 0.0
    plot_YF_max: Optional[float] = 1.0
    plot_YO_min: Optional[float] = 0.0
    plot_YO_max: Optional[float] = 0.25
    plot_YC2_min: Optional[float] = 0.0
    plot_YC2_max: Optional[float] = 1.0
    plot_YW_min: Optional[float] = 0.0
    plot_YW_max: Optional[float] = 1.0
    plot_RHO_min: Optional[float] = None
    plot_RHO_max: Optional[float] = None
    plot_U_min: Optional[float] = None
    plot_U_max: Optional[float] = None
    plot_V_min: Optional[float] = None
    plot_V_max: Optional[float] = None
    plot_P_min: Optional[float] = None
    plot_P_max: Optional[float] = None

    # Log-scale plots (LogNorm)
    plot_YC_min: Optional[float] = 1e-10
    plot_YC_max: Optional[float] = 1e-1
    plot_HRR_min: Optional[float] = 1e-1
    plot_HRR_max: Optional[float] = 1e9  # None -> auto from data
    plot_YNO_min: Optional[float] = 1e-8
    plot_YNO_max: Optional[float] = 1e-3


def parse_config() -> Config:
    cfg = Config()
    return cfg




