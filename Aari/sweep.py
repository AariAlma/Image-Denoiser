"""
sweep.py  —  Hyperparameter sweep for Primal-Dual Douglas-Rachford
=================================================================
Sweeps gamma × t × rho for both L1 and L2 problem variants.
Run from the project/Aari/ directory:

    python sweep.py

Outputs (written to project/Aari/):
    PDDR_Sweep_Results-l1.csv
    PDDR_Sweep_Results-l2.csv
    PDDR_Sweep_Heatmap-l1.png
    PDDR_Sweep_Heatmap-l2.png
"""

import csv
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

from pipeline import build_corrupted_image
from solver import primal_dual_dr_solve

# ---- Fixed corruption settings (match main.py defaults) ------------------
IMAGE_PATH    = "../testimages/testimages/cameraman.jpg"
BLUR_KIND     = "gaussian"
BLUR_KSIZE    = 9
BLUR_SIGMA    = 2.0
MOTION_LENGTH = 9
MOTION_ANGLE  = 0.0
NOISE_TYPE    = "gaussian"
NOISE_LEVEL   = 0.02
SEED          = 0
SCALE         = 1.0

# ---- Sweep settings -------------------------------------------------------
MAXITER  = 300      # reduced from 500 to keep sweep time reasonable
TOL      = 1e-4
TOP_N    = 5
PROBLEMS = ["l2", "l1"]

# ---- Parameter grids ------------------------------------------------------
GAMMA_VALS = [0.001, 0.005, 0.01, 0.05, 0.1]   # TV regularization weight
T_VALS     = [0.25, 0.5, 1.0, 1.5, 2.0]         # step size (must be > 0)
RHO_VALS   = [0.5, 1.0, 1.5]                     # relaxation (must be in (0, 2))

FIELDNAMES = ["gamma", "t", "rho", "PSNR", "objective", "iterations", "converged"]

# ---------------------------------------------------------------------------


def compute_psnr(x_sol, x_true):
    """PSNR in dB; assumes pixel values in [0, 1]."""
    mse = float(np.mean((x_sol - x_true) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def run_sweep(problem, x_true, b, psf):
    """Run the full gamma × t × rho grid for one problem type.
    Returns a list of result dicts."""
    combos = list(itertools.product(GAMMA_VALS, T_VALS, RHO_VALS))
    n_total = len(combos)
    print(f"\n{'='*60}")
    print(f"  problem={problem.upper()}  |  {n_total} combinations")
    print(f"  gamma: {GAMMA_VALS}")
    print(f"  t:     {T_VALS}")
    print(f"  rho:   {RHO_VALS}")
    print(f"{'='*60}")

    rows = []
    t_start = time.time()

    for idx, (gamma, t, rho) in enumerate(combos, 1):
        x_sol, info = primal_dual_dr_solve(
            b, psf,
            problem=problem,
            gamma=gamma,
            t=t,
            rho=rho,
            maxiter=MAXITER,
            tol=TOL,
            verbose=False,
        )
        psnr = compute_psnr(x_sol, x_true)
        rows.append({
            "gamma":      gamma,
            "t":          t,
            "rho":        rho,
            "PSNR":       psnr,
            "objective":  info["objective"][-1],
            "iterations": info["iterations"],
            "converged":  info["converged"],
        })

        if idx % 15 == 0 or idx == n_total:
            elapsed = time.time() - t_start
            print(f"  Completed {idx:3d} / {n_total}  ({elapsed:.1f}s elapsed)")

    elapsed_total = time.time() - t_start
    print(f"  Sweep done — {elapsed_total:.1f}s total")
    return rows


def save_csv(rows, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results saved → {filename}")


def print_top_n(rows, problem, n=TOP_N):
    sorted_rows = sorted(rows, key=lambda r: r["PSNR"], reverse=True)
    print(f"\n--- TOP {n} ({problem.upper()}) ---")
    header = f"{'gamma':>8}  {'t':>6}  {'rho':>5}  {'PSNR (dB)':>10}  {'iters':>6}  {'conv':>5}"
    print(header)
    print("-" * len(header))
    for r in sorted_rows[:n]:
        print(
            f"{r['gamma']:>8.4f}  {r['t']:>6.2f}  {r['rho']:>5.2f}"
            f"  {r['PSNR']:>10.4f}  {r['iterations']:>6d}  {str(r['converged']):>5}"
        )


def build_pivot(rows, rho, row_vals, col_vals, field="PSNR"):
    """Build a 2-D numpy array for the heatmap: rows=gamma, cols=t."""
    lookup = {(r["gamma"], r["t"]): r[field] for r in rows if r["rho"] == rho}
    grid = np.zeros((len(row_vals), len(col_vals)))
    for ri, g in enumerate(row_vals):
        for ci, t in enumerate(col_vals):
            grid[ri, ci] = lookup.get((g, t), float("nan"))
    return grid


def plot_heatmaps(rows, problem):
    n_rho = len(RHO_VALS)
    all_psnr = [r["PSNR"] for r in rows]
    vmin, vmax = min(all_psnr), max(all_psnr)

    fig, axes = plt.subplots(1, n_rho, figsize=(5 * n_rho, 4), sharey=True)
    if n_rho == 1:
        axes = [axes]

    for ax, rho in zip(axes, RHO_VALS):
        grid = build_pivot(rows, rho, GAMMA_VALS, T_VALS, field="PSNR")
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            vmin=vmin, vmax=vmax,
        )
        ax.set_xticks(range(len(T_VALS)))
        ax.set_xticklabels(T_VALS, fontsize=9)
        ax.set_yticks(range(len(GAMMA_VALS)))
        ax.set_yticklabels(GAMMA_VALS, fontsize=9)
        ax.set_xlabel("t (step size)")
        ax.set_ylabel("gamma (TV weight)")
        ax.set_title(f"rho = {rho}")
        plt.colorbar(im, ax=ax, label="PSNR (dB)")

    fig.suptitle(
        f"PDDR Hyperparameter Sweep — PSNR (dB)  [{problem.upper()}]",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fname = f"PDDR_Sweep_Heatmap-{problem}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {fname}")


def main():
    # ---- Load and corrupt image once (shared across both sweeps) ----------
    print("Loading and corrupting image...")
    x_true, b, psf = build_corrupted_image(
        IMAGE_PATH,
        blur_kind=BLUR_KIND,
        blur_ksize=BLUR_KSIZE,
        blur_sigma=BLUR_SIGMA,
        motion_length=MOTION_LENGTH,
        motion_angle=MOTION_ANGLE,
        noise_type=NOISE_TYPE,
        noise_level=NOISE_LEVEL,
        scale=SCALE,
        seed=SEED,
    )
    print(f"  Image shape: {x_true.shape}")

    # ---- Run sweep for each problem variant --------------------------------
    for problem in PROBLEMS:
        rows = run_sweep(problem, x_true, b, psf)
        save_csv(rows, f"PDDR_Sweep_Results-{problem}.csv")
        print_top_n(rows, problem)
        plot_heatmaps(rows, problem)

    print("\nDone.")


if __name__ == "__main__":
    main()
