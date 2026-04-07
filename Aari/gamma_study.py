"""
gamma_study.py  —  Effect of gamma on PSNR across noise levels (PDDR)
=====================================================================
Mirrors Daniel's gammastudy.m but uses the Primal-Dual DR solver.

Run from the project/Aari/ directory:
    python gamma_study.py

Output:
    PDDR_GammaStudy.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from pipeline import build_corrupted_image


def _next_path(stem, ext):
    """Return stem_1.ext, stem_2.ext, … — first index not already on disk."""
    i = 1
    while os.path.exists(f"{stem}_{i}.{ext}"):
        i += 1
    return f"{stem}_{i}.{ext}"
from solver import primal_dual_dr_solve

# ---- Settings ----------------------------------------------------------------
IMAGE_PATH   = "../testimages/testimages/cameraman.jpg"
BLUR_KIND    = "gaussian"
BLUR_KSIZE   = 9
BLUR_SIGMA   = 2.0
NOISE_TYPE   = "gaussian"
SEED         = 0

PROBLEM      = "l1"
T            = 0.25      # best L1 step size from sweep
RHO          = 0.5       # best L1 relaxation from sweep
MAXITER      = 500
TOL          = 1e-4

GAMMA_VALS   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2]
# ------------------------------------------------------------------------------


def psnr(x_sol, x_true):
    mse = float(np.mean((x_sol - x_true) ** 2))
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def main():
    psnr_grid = np.zeros((len(NOISE_LEVELS), len(GAMMA_VALS)))

    for ni, noise_level in enumerate(NOISE_LEVELS):
        print(f"\nnoise_level = {noise_level:.2f}")
        x_true, b, psf = build_corrupted_image(
            IMAGE_PATH,
            blur_kind=BLUR_KIND, blur_ksize=BLUR_KSIZE, blur_sigma=BLUR_SIGMA,
            noise_type=NOISE_TYPE, noise_level=noise_level,
            seed=SEED,
        )

        for gi, gamma in enumerate(GAMMA_VALS):
            x_sol, _ = primal_dual_dr_solve(
                b, psf,
                problem=PROBLEM,
                gamma=gamma,
                t=T, rho=RHO,
                maxiter=MAXITER, tol=TOL,
                verbose=False,
            )
            p = psnr(x_sol, x_true)
            psnr_grid[ni, gi] = p
            print(f"  gamma={gamma:.3f}  PSNR={p:.2f} dB")

    # ---- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for ni, noise_level in enumerate(NOISE_LEVELS):
        ax.semilogx(
            GAMMA_VALS, psnr_grid[ni],
            "o-", linewidth=1.5,
            label=f"noise = {noise_level:.2f}",
        )
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(r"Effect of $\gamma$ on Reconstruction Quality (PDDR, $\ell_1$)")
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fname = _next_path("PDDR_GammaStudy", "png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {fname}")
    plt.show()


if __name__ == "__main__":
    main()
