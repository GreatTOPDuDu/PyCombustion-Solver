"""
Thin wrapper to run the modular CBm0 solver.
"""
from __future__ import annotations

import sys

from CBm0.config import parse_config
from CBm0.main import main


if __name__ == '__main__':
    print("Numba parallel JIT enabled. First run may be slower (compiling...)")
    cfg = parse_config()
    sys.exit(main(cfg))
