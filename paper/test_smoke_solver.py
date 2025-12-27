import tempfile

from CBm0.config import Config, apply_mode_defaults
from CBm0.main import main


def test_small_inert_run_smoke(tmp_path):
    """End-to-end smoke test of the main CBm0 solver on a tiny inert case.

    This keeps the grid and final time small so that the test
    runs quickly, while still exercising the full time-stepping
    loop, pressure projection, and I/O pipeline.
    """

    # Configure a very small, fast inert case
    cfg = Config()
    cfg.Nx = 32
    cfg.Ny = 64
    cfg.Lx = 0.02
    cfg.Ly = 0.04
    cfg.mode = "inert"  # disable chemistry for speed
    cfg.t_final = 0.01
    cfg.cfl_adv = 0.5
    cfg.cfl_diff = 0.5

    # Direct all output into pytest's temporary directory
    cfg.output_base_dir = str(tmp_path)
    cfg.output_method_dir = "smoke"

    cfg = apply_mode_defaults(cfg)
    ret = main(cfg)

    assert ret == 0
