from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class Config:
    # grid
    Nx: int = 201
    Ny: int = 401
    Lx: float = 0.2
    Ly: float = 0.4

    # thermodynamic reference
    P0_pa: float = 1.0e5

    # inlets
    v_fuel: float = 0.5
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
    t_final: float = 2.0
    cfl_adv: float = 0.35
    cfl_diff: float = 0.35
    # Save condition: If None, that criterion is disabled
    save_every: Optional[int] = 50        # Step-based save interval (None -> disabled)
    save_interval: Optional[float] = None # Physical time interval save (None -> disabled)

    # pressure solver (multigrid)
    mg_pre_smooth: int = 4
    mg_post_smooth: int = 4
    mg_cycles_per_step: int = 2
    mg_coarsest_min: int = 25

    # transport
    Pr_mix_ref: float = 0.7
    adv_limiter: str = "superbee"  # 'minmod'|'vanleer'|'superbee'

    # chemistry
    kinetics_model: str = "wd2"   # 'wd2' | 'h2'
    CHEM_SUBSTEPS: int = 25
    fuel_type: str = "CH4"        # 'CH4' | 'H2'
    enable_thermal_NOx: bool = True

    # physics / runtime mode
    # mode controls high-level physics fidelity; derived flags below can be overridden explicitly.
    # reference: analytic Shomate thermo + full transport + chemistry on
    # fast:      LUT thermo / simplified transport (Le=1) + chemistry on
    # inert:     thermo/transport as selected, but chemistry disabled
    mode: str = "reference"       # 'reference' | 'fast' | 'inert'

    # derived toggles (set by `mode`, but may be overridden in configs if needed)
    use_thermo_lut: bool = True   # True => use cp/h LUTs instead of direct Shomate evaluation
    transport_model: str = "full"  # 'full' | 'Le1'
    chemistry_on: bool = True      # False => skip chemistry source terms entirely

    # Arrhenius multipliers (global knobs for teaching/sensitivity studies)
    # Westbrook–Dryer CH4 mechanism
    wd2_A_mult: float = 1.0        # multiplies both A1 and A2
    wd2_Ea_mult: float = 1.0       # multiplies both Ea1 and Ea2

    # Global H2 mechanism
    h2_A_mult: float = 1.0
    h2_Ea_mult: float = 1.0

    # ignition
    ignition_mode: str = "off"     

    # ignition detection
    ign_hrr_threshold_Wm: float = 10.0

    # parallelization
    num_threads: int = 4  

    # output directories
    output_base_dir: str = 'out1'
    output_method_dir: str = 'test1'

    # lightweight in-situ monitoring
    # monitor_interval <= 0 disables additional logging.
    monitor_interval: int = 50
    monitor_centerline: bool = True

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
    return apply_mode_defaults(cfg)


def apply_mode_defaults(cfg: Config) -> Config:
    """Apply high-level physics defaults based on cfg.mode.

    This keeps the "reference / fast / inert" presets in one place so that
    documentation and examples can simply refer to a single `mode` switch.
    """

    mode = (cfg.mode or "reference").lower()

    if mode == "reference":
        cfg.use_thermo_lut = False
        cfg.transport_model = "full"
        cfg.chemistry_on = True
    elif mode == "fast":
        cfg.use_thermo_lut = True
        cfg.transport_model = "Le1"
        cfg.chemistry_on = True
    elif mode == "inert":
        # Inert mode disables chemistry; thermo/transport remain configurable
        cfg.chemistry_on = False
        # Keep existing thermo/transport settings unless the user overrides them
    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}; expected 'reference', 'fast', or 'inert'.")

    return cfg








