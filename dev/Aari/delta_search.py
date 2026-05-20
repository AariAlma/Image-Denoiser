"""
delta_search.py  —  Golden section search for optimal Huber delta (PDDR)
=========================================================================
The Huber loss phi_delta combines L1 (large residuals) and L2 (small
residuals) via a threshold parameter delta.  This script finds the delta
that maximises PSNR for each noise level using golden section search,
with gamma fixed to a sensible default per noise type.

Run from the project/Aari/ directory:
    python delta_search.py

Outputs (auto-numbered):
    PDDR_DeltaSearch_evaluations_N.png  — evaluated points per combo
    PDDR_DeltaSearch_optimal_N.png      — optimal delta & PSNR vs noise level
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
NOISE_LEVELS  = [0.01, 0.05, 0.15, 0.3, 0.5]

# Gamma fixed per noise type (use values found by gamma_search.py, or defaults)
GAMMA_FIXED   = {"gaussian": 0.05, "saltpepper": 0.1}

T             = 0.25
RHO           = 0.5
MAXITER       = 500
TOL           = 1e-4
USE_GPU       = True

# Search bounds for delta (log-space; delta is in image intensity units [0,1])
LOG_DELTA_MIN = math.log(1e-3)   # delta = 0.001
LOG_DELTA_MAX = math.log(0.5)    # delta = 0.5
SEARCH_TOL    = 0.05             # stop when log-space interval width < this
# ------------------------------------------------------------------------------

_PHI = (math.sqrt(5) - 1) / 2

NOISE_LABELS = {"gaussian": "Gaussian", "saltpepper": "Salt & Pepper"}
COMBO_STYLES = {
    "gaussian":   dict(color="tab:blue",   linestyle="-",  marker="o"),
    "saltpepper": dict(color="tab:orange", linestyle="-",  marker="o"),
}


def _psnr(x_sol, x_true):
    mse = float(np.mean((x_sol - x_true) ** 2))
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def _evaluate(log_delta, b, psf, x_true, gamma):
    delta = math.exp(log_delta)
    x_sol, _ = primal_dual_dr_solve(
        b, psf,
        problem="huber", gamma=gamma,
        t=T, rho=RHO,
        huber_delta=delta,
        maxiter=MAXITER, tol=TOL,
        verbose=False, use_gpu=USE_GPU,
    )
    return _psnr(x_sol, x_true)


def golden_section_search(b, psf, x_true, gamma):
    """
    Maximise PSNR(delta) over delta in exp([LOG_DELTA_MIN, LOG_DELTA_MAX]).

    Returns
    -------
    best_delta : float
    best_psnr  : float
    history    : list of (delta, psnr) for every evaluation made
    """
    lo, hi = LOG_DELTA_MIN, LOG_DELTA_MAX

    c = hi - _PHI * (hi - lo)
    d = lo + _PHI * (hi - lo)

    fc = _evaluate(c, b, psf, x_true, gamma)
    fd = _evaluate(d, b, psf, x_true, gamma)
    history = [(math.exp(c), fc), (math.exp(d), fd)]

    step = 2
    print(f"    step  1  δ={math.exp(c):.5f}  PSNR={fc:.3f} dB")
    print(f"    step  2  δ={math.exp(d):.5f}  PSNR={fd:.3f} dB")

    while hi - lo > SEARCH_TOL:
        step += 1
        if fc > fd:
            hi = d
            d, fd = c, fc
            c = hi - _PHI * (hi - lo)
            fc = _evaluate(c, b, psf, x_true, gamma)
            history.append((math.exp(c), fc))
            print(f"    step {step:2d}  δ={math.exp(c):.5f}  PSNR={fc:.3f} dB")
        else:
            lo = c
            c, fc = d, fd
            d = lo + _PHI * (hi - lo)
            fd = _evaluate(d, b, psf, x_true, gamma)
            history.append((math.exp(d), fd))
            print(f"    step {step:2d}  δ={math.exp(d):.5f}  PSNR={fd:.3f} dB")

    best_delta = math.exp((lo + hi) / 2)
    best_psnr  = max(fc, fd)
    return best_delta, best_psnr, history


def main():
    combo_results = {}   # noise_type → [(noise_level, best_delta, best_psnr, history)]

    for noise_type in NOISE_TYPES:
        gamma = GAMMA_FIXED[noise_type]
        print(f"\n{'='*60}")
        print(f"  noise_type={NOISE_LABELS[noise_type]}  |  gamma={gamma}  |  problem=HUBER")
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
            best_delta, best_p, history = golden_section_search(b, psf, x_true, gamma)
            print(f"  → Optimal δ = {best_delta:.5f}   PSNR = {best_p:.3f} dB")
            results.append((noise_level, best_delta, best_p, history))

        combo_results[noise_type] = results

        print(f"\n  {'Noise':>8}  {'Optimal δ':>12}  {'PSNR (dB)':>10}")
        print(f"  {'-'*8}  {'-'*12}  {'-'*10}")
        for nl, bd, bp, _ in results:
            print(f"  {nl:>8.2f}  {bd:>12.5f}  {bp:>10.3f}")

    # ---- Save CSV ------------------------------------------------------------
    csv_fname = _next_path("PDDR_DeltaSearch_optimal", "csv")
    with open(csv_fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_type", "noise_level", "gamma_fixed", "optimal_delta", "psnr_db"])
        writer.writeheader()
        for noise_type in NOISE_TYPES:
            for nl, bd, bp, _ in combo_results[noise_type]:
                writer.writerow({
                    "noise_type":    noise_type,
                    "noise_level":   nl,
                    "gamma_fixed":   GAMMA_FIXED[noise_type],
                    "optimal_delta": round(bd, 6),
                    "psnr_db":       round(bp, 4),
                })
    print(f"\nSaved → {csv_fname}")

    # ---- Plot 1: evaluation scatter per noise type ---------------------------
    fig1, axes1 = plt.subplots(1, len(NOISE_TYPES), figsize=(6 * len(NOISE_TYPES), 4))
    for ci, noise_type in enumerate(NOISE_TYPES):
        ax = axes1[ci]
        for noise_level, best_delta, _, history in combo_results[noise_type]:
            deltas = [d for d, _ in history]
            psnrs  = [p for _, p in history]
            ax.scatter(deltas, psnrs, s=40,
                       label=f"noise={noise_level:.2f}  δ*={best_delta:.4f}")
            ax.axvline(best_delta, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\delta$ (Huber threshold)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"{NOISE_LABELS[noise_type]} noise  |  Huber", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig1.suptitle(r"Golden Section Search for Huber $\delta$  (PDDR)", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fname1 = _next_path("PDDR_DeltaSearch_evaluations", "png")
    fig1.savefig(fname1, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {fname1}")

    # ---- Plot 2: optimal delta & PSNR vs noise level -------------------------
    fig2, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(12, 5))

    for noise_type in NOISE_TYPES:
        results = combo_results[noise_type]
        nls    = [r[0] for r in results]
        deltas = [r[1] for r in results]
        psnrs  = [r[2] for r in results]
        style  = COMBO_STYLES[noise_type]
        label  = NOISE_LABELS[noise_type]

        ax_d.plot(nls, deltas, linewidth=1.8, markersize=6, label=label, **style)
        ax_p.plot(nls, psnrs,  linewidth=1.8, markersize=6, label=label, **style)

    ax_d.set_xlabel("Noise level")
    ax_d.set_ylabel(r"Optimal $\delta$")
    ax_d.set_title(r"Optimal Huber $\delta$ vs Noise Level")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, linestyle="--", alpha=0.4)

    ax_p.set_xlabel("Noise level")
    ax_p.set_ylabel("Best PSNR (dB)")
    ax_p.set_title("Best PSNR vs Noise Level  (Huber)")
    ax_p.legend(fontsize=8)
    ax_p.grid(True, linestyle="--", alpha=0.4)

    fig2.suptitle(
        r"Optimal Huber $\delta$ and PSNR vs Noise Level  (PDDR)",
        fontsize=13, fontweight="bold",
    )
    fig2.tight_layout()
    fname2 = _next_path("PDDR_DeltaSearch_optimal", "png")
    fig2.savefig(fname2, dpi=120, bbox_inches="tight")
    print(f"Saved → {fname2}")

    plt.show()


if __name__ == "__main__":
    main()
