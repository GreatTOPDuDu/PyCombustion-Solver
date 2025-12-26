import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "cavity_output")
FIG_DIR = os.path.join(HERE, "validation_figs")
REF_DIR = os.path.join(HERE, "cavity_reference")
os.makedirs(FIG_DIR, exist_ok=True)


def load_reference_centerlines(re):
    ref_u_file = os.path.join(REF_DIR, f"Re{re}_u.csv")
    ref_v_file = os.path.join(REF_DIR, f"Re{re}_v.csv")

    if not (os.path.exists(ref_u_file) and os.path.exists(ref_v_file)):
        raise FileNotFoundError(
            f"Reference files for Re={re} not found in {REF_DIR}."
        )

    y_ref, u_ref = np.loadtxt(ref_u_file, delimiter=",", skiprows=1).T
    x_ref, v_ref = np.loadtxt(ref_v_file, delimiter=",", skiprows=1).T

    return x_ref, y_ref, u_ref, v_ref


def main():
    u_file = os.path.join(DATA_DIR, "u_centerline.csv")
    v_file = os.path.join(DATA_DIR, "v_centerline.csv")

    if not (os.path.exists(u_file) and os.path.exists(v_file)):
        raise FileNotFoundError(
            "Run examples/run_cavity.py first to generate centerline CSV files."
        )

    cfg_file = os.path.join(HERE, "cavity_config.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    re_case = int(cfg.get("Re", 100))

    y_num, u_num = np.loadtxt(u_file, delimiter=",", skiprows=1).T
    x_num, v_num = np.loadtxt(v_file, delimiter=",", skiprows=1).T

    x_ref, y_ref, u_ref, v_ref = load_reference_centerlines(re_case)

    # Interpolate numerical solution onto reference coordinates
    u_num_on_ref = np.interp(y_ref, y_num, u_num)
    v_num_on_ref = np.interp(x_ref, x_num, v_num)

    # Compute errors
    u_err = u_num_on_ref - u_ref
    v_err = v_num_on_ref - v_ref

    u_rel_max = np.max(np.abs(u_err)) / max(1e-8, np.max(np.abs(u_ref))) * 100.0
    v_rel_max = np.max(np.abs(v_err)) / max(1e-8, np.max(np.abs(v_ref))) * 100.0

    print(f"Max relative error in u centerline (Re={re_case}): {u_rel_max:.2f} %")
    print(f"Max relative error in v centerline (Re={re_case}): {v_rel_max:.2f} %")

    # Plot comparison
    plt.figure(figsize=(5, 6))
    plt.plot(u_num, y_num, label="CBm0", lw=2)
    plt.plot(u_ref, y_ref, "o", label="Ghia et al.")
    plt.xlabel("u")
    plt.ylabel("y")
    plt.title(f"Lid-driven cavity: u centerline (x=0.5, Re={re_case})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cavity_u_centerline.png"), dpi=300)

    plt.figure(figsize=(5, 6))
    plt.plot(x_num, v_num, label="CBm0", lw=2)
    plt.plot(x_ref, v_ref, "o", label="Ghia et al.")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(f"Lid-driven cavity: v centerline (y=0.5, Re={re_case})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cavity_v_centerline.png"), dpi=300)

    print(f"Figures written to {FIG_DIR}.")


if __name__ == "__main__":
    main()
