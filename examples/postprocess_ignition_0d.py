import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "ignition_output")
FIG_DIR = os.path.join(HERE, "validation_figs")
os.makedirs(FIG_DIR, exist_ok=True)


def load_history(path: str):
    t, T, YF = np.loadtxt(path, delimiter=",", skiprows=1).T
    return t, T, YF


def ignition_delay(t: np.ndarray, T: np.ndarray, dT_threshold: float = 400.0) -> float | None:
    T0 = T[0]
    mask = T - T0 > dT_threshold
    if not np.any(mask):
        return None
    idx = np.argmax(mask)
    return float(t[idx])


def main():
    coarse_path = os.path.join(DATA_DIR, "ignition_0d_history.csv")
    ref_path = os.path.join(DATA_DIR, "ignition_0d_history_ref.csv")

    if not os.path.exists(coarse_path):
        raise FileNotFoundError(
            "Run examples/run_ignition_0d.py first to generate ignition_0d_history.csv."
        )
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            "Generate a reference solution by re-running run_ignition_0d.py "
            "with a much smaller time step and save it as "
            "ignition_0d_history_ref.csv in the same directory."
        )

    t_c, T_c, YF_c = load_history(coarse_path)
    t_r, T_r, YF_r = load_history(ref_path)

    # Interpolate reference onto coarse time grid for error metrics
    T_r_on_c = np.interp(t_c, t_r, T_r)

    err_T = T_c - T_r_on_c
    rel_L2 = np.sqrt(np.mean(err_T**2)) / max(1e-8, np.max(np.abs(T_r_on_c))) * 100.0

    tign_c = ignition_delay(t_c, T_c)
    tign_r = ignition_delay(t_r, T_r)

    if tign_c is not None and tign_r is not None:
        tign_err = abs(tign_c - tign_r) / max(1e-8, tign_r) * 100.0
        print(f"Ignition delay (coarse)  = {tign_c:.4e} s")
        print(f"Ignition delay (ref)     = {tign_r:.4e} s")
        print(f"Relative error in t_ign  = {tign_err:.2f} %")
    else:
        print("Ignition not detected in one of the histories; cannot compute t_ign error.")

    print(f"Relative L2 error in T(t): {rel_L2:.2f} %")

    # Plot temperature histories
    plt.figure(figsize=(6, 4))
    plt.plot(t_c, T_c, label="coarse dt")
    plt.plot(t_r, T_r, "--", label="reference dt")
    plt.xlabel("t [s]")
    plt.ylabel("T [K]")
    plt.title("0D ignition: temperature history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ignition_T_history.png"), dpi=300)
    print(f"Figure written to {FIG_DIR}.")


if __name__ == "__main__":
    main()
