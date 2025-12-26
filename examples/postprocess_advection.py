import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "advection_output")
FIG_DIR = os.path.join(HERE, "validation_figs")
os.makedirs(FIG_DIR, exist_ok=True)

LIMITERS = ["minmod", "superbee", "vanleer"]


def interface_width(x: np.ndarray, phi: np.ndarray, low: float = 0.1, high: float = 0.9) -> int:
    mask = (phi >= low) & (phi <= high)
    return int(np.count_nonzero(mask))


def main():
    plt.figure(figsize=(6, 4))

    for limiter in LIMITERS:
        path = os.path.join(DATA_DIR, f"top_hat_{limiter}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run examples/run_advection_1d.py first."
            )

        x, phi = np.loadtxt(path, delimiter=",", skiprows=1).T

        # Basic monotonicity / overshoot check
        phi_min = float(np.min(phi))
        phi_max = float(np.max(phi))
        width = interface_width(x, phi)

        print(f"Limiter={limiter:7s}: min(phi)={phi_min:.3f}, max(phi)={phi_max:.3f}, "
              f"interface width (0.1-0.9)={width} cells")

        plt.plot(x, phi, label=f"{limiter}")

    plt.xlabel("x")
    plt.ylabel("phi")
    plt.title("Top-hat advection: TVD limiter comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "advection_top_hat_limiters.png"), dpi=300)
    print(f"Figure written to {FIG_DIR}.")


if __name__ == "__main__":
    main()
