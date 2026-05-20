"""
gamma_search.py  —  Golden section search for optimal gamma (PDDR)
===================================================================
Finds the gamma that maximises PSNR using golden section search for every
combination of noise type (gaussian / saltpepper) and problem (l1 / l2).
The search is performed in log-space since gamma spans several orders of
magnitude, and PSNR vs gamma is unimodal (rises then falls).

Run from the project/Aari/ directory:
    python gamma_search.py

Outputs (auto-numbered so previous runs are never overwritten):
    PDDR_GammaSearch_evaluations_N.png  — evaluated points per combination
    PDDR_GammaSearch_optimal_N.png      — optimal gamma & PSNR vs noise level
"""

import csv
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from pipeline import build_corrupted_image
from solver import primal_dual_dr_solve


def _next_path(stem, ext):
    """Return stem_1.ext, stem_2.ext, … — first index not already on disk."""
    i = 1
    while os.path.exists(f"{stem}_{i}.{ext}"):
        i += 1
    return f"{stem}_{i}.{ext}"


# ---- Settings ----------------------------------------------------------------
IMAGE_PATH    = "../testimages/testimages/cameraman.jpg"
BLUR_KIND     = "gaussian"
BLUR_KSIZE    = 9
BLUR_SIGMA    = 2.0
SEED          = 0

NOISE_TYPES   = ["gaussian", "saltpepper"]
PROBLEMS      = ["l1", "l2", "huber"]

# Per-problem solver params (from sweep results)
SOLVER_PARAMS = {
    "l1":    {"t": 0.25, "rho": 0.5},
    "l2":    {"t": 1.0,  "rho": 1.0},
    "huber": {"t": 0.25, "rho": 0.5, "delta": 0.1},
}

MAXITER       = 500
TOL           = 1e-4
USE_GPU       = True    # set False to force CPU
NOISE_LEVELS  = [0.01, 0.05, 0.15, 0.3, 0.5]

# Search bounds for gamma (log-space)
LOG_GAMMA_MIN = math.log(1e-4)   # gamma = 0.0001
LOG_GAMMA_MAX = math.log(2.0)    # gamma = 2.0
SEARCH_TOL    = 0.05             # stop when log-space interval width < this
# ------------------------------------------------------------------------------

_PHI = (math.sqrt(5) - 1) / 2   # golden ratio conjugate ≈ 0.618

NOISE_LABELS = {"gaussian": "Gaussian", "saltpepper": "Salt & Pepper"}
COMBO_STYLES = {
    ("gaussian",   "l1"):    dict(color="tab:blue",   linestyle="-",  marker="o"),
    ("gaussian",   "l2"):    dict(color="tab:blue",   linestyle="--", marker="s"),
    ("gaussian",   "huber"): dict(color="tab:blue",   linestyle=":",  marker="^"),
    ("saltpepper", "l1"):    dict(color="tab:orange", linestyle="-",  marker="o"),
    ("saltpepper", "l2"):    dict(color="tab:orange", linestyle="--", marker="s"),
    ("saltpepper", "huber"): dict(color="tab:orange", linestyle=":",  marker="^"),
}


def _psnr(x_sol, x_true):
    mse = float(np.mean((x_sol - x_true) ** 2))
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def _evaluate(log_gamma, b, psf, x_true, problem):
    gamma = math.exp(log_gamma)
    sp = SOLVER_PARAMS[problem]
    x_sol, _ = primal_dual_dr_solve(
        b, psf,
        problem=problem, gamma=gamma,
        t=sp["t"], rho=sp["rho"],
        huber_delta=sp.get("delta", 0.1),
        maxiter=MAXITER, tol=TOL,
        verbose=False, use_gpu=USE_GPU,
    )
    return _psnr(x_sol, x_true)


def golden_section_search(b, psf, x_true, problem):
    """
    Maximise PSNR(gamma) over gamma in exp([LOG_GAMMA_MIN, LOG_GAMMA_MAX]).

    Returns
    -------
    best_gamma : float
    best_psnr  : float
    history    : list of (gamma, psnr) for every evaluation made
    """
    lo, hi = LOG_GAMMA_MIN, LOG_GAMMA_MAX

    c = hi - _PHI * (hi - lo)
    d = lo + _PHI * (hi - lo)

    fc = _evaluate(c, b, psf, x_true, problem)
    fd = _evaluate(d, b, psf, x_true, problem)
    history = [(math.exp(c), fc), (math.exp(d), fd)]

    step = 2
    print(f"    step  1  γ={math.exp(c):.5f}  PSNR={fc:.3f} dB")
    print(f"    step  2  γ={math.exp(d):.5f}  PSNR={fd:.3f} dB")

    while hi - lo > SEARCH_TOL:
        step += 1
        if fc > fd:
            hi = d
            d, fd = c, fc
            c = hi - _PHI * (hi - lo)
            fc = _evaluate(c, b, psf, x_true, problem)
            history.append((math.exp(c), fc))
            print(f"    step {step:2d}  γ={math.exp(c):.5f}  PSNR={fc:.3f} dB")
        else:
            lo = c
            c, fc = d, fd
            d = lo + _PHI * (hi - lo)
            fd = _evaluate(d, b, psf, x_true, problem)
            history.append((math.exp(d), fd))
            print(f"    step {step:2d}  γ={math.exp(d):.5f}  PSNR={fd:.3f} dB")

    best_gamma = math.exp((lo + hi) / 2)
    best_psnr  = max(fc, fd)
    return best_gamma, best_psnr, history


def main():
    # combo_results[(noise_type, problem)] = [(noise_level, best_gamma, best_psnr, history), ...]
    combo_results = {}

    for noise_type in NOISE_TYPES:
        for problem in PROBLEMS:
            key = (noise_type, problem)
            print(f"\n{'='*60}")
            print(f"  noise_type={NOISE_LABELS[noise_type]}  |  problem={problem.upper()}")
            print(f"{'='*60}")

            results = []
            for noise_level in NOISE_LEVELS:
                print(f"\n  noise_level = {noise_level:.2f}")
                x_true, b, psf = build_corrupted_image(
                    IMAGE_PATH,
                    blur_kind=BLUR_KIND, blur_ksize=BLUR_KSIZE, blur_sigma=BLUR_SIGMA,
                    noise_type=noise_type, noise_level=noise_level,
                    seed=SEED,
                )
                best_gamma, best_p, history = golden_section_search(b, psf, x_true, problem)
                print(f"  → Optimal γ = {best_gamma:.5f}   PSNR = {best_p:.3f} dB")
                results.append((noise_level, best_gamma, best_p, history))

            combo_results[key] = results

            # Per-combo summary
            print(f"\n  {'Noise':>8}  {'Optimal γ':>12}  {'PSNR (dB)':>10}")
            print(f"  {'-'*8}  {'-'*12}  {'-'*10}")
            for nl, bg, bp, _ in results:
                print(f"  {nl:>8.2f}  {bg:>12.5f}  {bp:>10.3f}")

    # ---- Plot 1: 2×2 grid of evaluation scatter plots -----------------------
    # ---- Save CSV ------------------------------------------------------------
    csv_fname = _next_path("PDDR_GammaSearch_optimal", "csv")
    with open(csv_fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_type", "problem", "noise_level", "optimal_gamma", "psnr_db"])
        writer.writeheader()
        for noise_type in NOISE_TYPES:
            for problem in PROBLEMS:
                for nl, bg, bp, _ in combo_results[(noise_type, problem)]:
                    writer.writerow({
                        "noise_type":    noise_type,
                        "problem":       problem,
                        "noise_level":   nl,
                        "optimal_gamma": round(bg, 6),
                        "psnr_db":       round(bp, 4),
                    })
    print(f"Saved → {csv_fname}")

    fig1, axes1 = plt.subplots(
        len(PROBLEMS), len(NOISE_TYPES),
        figsize=(6 * len(NOISE_TYPES), 4 * len(PROBLEMS)),
        sharex=True,
    )

    for ri, problem in enumerate(PROBLEMS):
        for ci, noise_type in enumerate(NOISE_TYPES):
            ax = axes1[ri, ci]
            results = combo_results[(noise_type, problem)]
            for noise_level, best_gamma, _, history in results:
                gammas = [g for g, _ in history]
                psnrs  = [p for _, p in history]
                ax.scatter(gammas, psnrs, s=40,
                           label=f"noise={noise_level:.2f}  γ*={best_gamma:.4f}")
                ax.axvline(best_gamma, linestyle="--", linewidth=0.8, alpha=0.35)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\gamma$")
            ax.set_ylabel("PSNR (dB)")
            ax.set_title(f"{NOISE_LABELS[noise_type]} noise  |  {problem.upper()}", fontsize=10)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig1.suptitle(r"Golden Section Search — Evaluated Points  (PDDR)", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fname1 = _next_path("PDDR_GammaSearch_evaluations", "png")
    fig1.savefig(fname1, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {fname1}")

    # ---- Plot 2: Optimal gamma vs noise level (all 4 combos on one plot) ----
    fig2, (ax_g, ax_p) = plt.subplots(1, 2, figsize=(13, 5))

    for noise_type in NOISE_TYPES:
        for problem in PROBLEMS:
            key = (noise_type, problem)
            results = combo_results[key]
            nls    = [r[0] for r in results]
            gammas = [r[1] for r in results]
            psnrs  = [r[2] for r in results]
            style  = COMBO_STYLES[key]
            label  = f"{NOISE_LABELS[noise_type]} / {problem.upper()}"

            ax_g.plot(nls, gammas, linewidth=1.8, markersize=6, label=label, **style)
            ax_p.plot(nls, psnrs,  linewidth=1.8, markersize=6, label=label, **style)

    ax_g.set_xlabel("Noise level")
    ax_g.set_ylabel(r"Optimal $\gamma$")
    ax_g.set_title(r"Optimal $\gamma$ vs Noise Level")
    ax_g.legend(fontsize=8)
    ax_g.grid(True, linestyle="--", alpha=0.4)

    ax_p.set_xlabel("Noise level")
    ax_p.set_ylabel("Best PSNR (dB)")
    ax_p.set_title("Best PSNR vs Noise Level")
    ax_p.legend(fontsize=8)
    ax_p.grid(True, linestyle="--", alpha=0.4)

    fig2.suptitle(
        r"Optimal $\gamma$ and PSNR vs Noise Level  (PDDR)",
        fontsize=13, fontweight="bold",
    )
    fig2.tight_layout()
    fname2 = _next_path("PDDR_GammaSearch_optimal", "png")
    fig2.savefig(fname2, dpi=120, bbox_inches="tight")
    print(f"Saved → {fname2}")

    plt.show()


if __name__ == "__main__":
    main()
