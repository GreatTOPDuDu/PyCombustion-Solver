from __future__ import annotations

import sys

from .config import parse_config
from .main import main


def _run():
	print("Numba parallel JIT enabled. First run may be slower (compiling...)")
	cfg = parse_config()
	return main(cfg)


if __name__ == "__main__":
	sys.exit(_run())
