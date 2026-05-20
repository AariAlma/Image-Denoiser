"""
gamma_varying.py  —  Image grid: gamma vs noise level (PDDR)
=============================================================
Mirrors Daniel's gammavarying.m but uses the Primal-Dual DR solver.
Rows = noise levels, columns = corrupted image + one column per gamma value.

Run from the project/Aari/ directory:
    python gamma_varying.py

Output:
    PDDR_GammaVarying.png
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
NOISE_TYPE   = "saltpepper"
SEED         = 0

PROBLEM      = "l1"
T            = 0.25      # best L1 step size from sweep
RHO          = 0.5       # best L1 relaxation from sweep
MAXITER      = 500
TOL          = 1e-4

GAMMA_VALS   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
NOISE_LEVELS = [0.01, 0.05, 0.15, 0.3, 0.5]
# ------------------------------------------------------------------------------


def psnr(x_sol, x_true):
    mse = float(np.mean((x_sol - x_true) ** 2))
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def main():
    nG = len(GAMMA_VALS)
    nN = len(NOISE_LEVELS)

    restored  = [[None] * nG for _ in range(nN)]
    corrupted = [None] * nN
    psnr_grid = np.zeros((nN, nG))

    for ni, noise_level in enumerate(NOISE_LEVELS):
        print(f"\nnoise_level = {noise_level:.2f}")
        x_true, b, psf = build_corrupted_image(
            IMAGE_PATH,
            blur_kind=BLUR_KIND, blur_ksize=BLUR_KSIZE, blur_sigma=BLUR_SIGMA,
            noise_type=NOISE_TYPE, noise_level=noise_level,
            seed=SEED,
        )
        corrupted[ni] = b

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
            restored[ni][gi] = x_sol
            psnr_grid[ni, gi] = p
            print(f"  gamma={gamma:.3f}  PSNR={p:.2f} dB")

    # ---- Plot ----------------------------------------------------------------
    ncols = nG + 1  # corrupted column + one per gamma
    fig, axes = plt.subplots(
        nN, ncols,
        figsize=(2.5 * ncols, 2.8 * nN),
    )

    for ni in range(nN):
        # Corrupted column
        ax0 = axes[ni, 0]
        ax0.imshow(corrupted[ni], cmap="gray", vmin=0, vmax=1)
        ax0.axis("off")
        if ni == 0:
            ax0.set_title("Corrupted", fontsize=9)
        # Row label: noise level on the left edge (set_ylabel is hidden by axis("off"))
        ax0.text(
            -0.05, 0.5, f"noise = {NOISE_LEVELS[ni]:.2f}",
            transform=ax0.transAxes,
            fontsize=9, fontweight="bold",
            va="center", ha="right", rotation=90,
        )

        for gi in range(nG):
            ax = axes[ni, gi + 1]
            ax.imshow(restored[ni][gi], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if ni == 0:
                ax.set_title(rf"$\gamma$ = {GAMMA_VALS[gi]:.3f}", fontsize=9)
            # PSNR below each image
            ax.text(
                0.5, -0.04, f"{psnr_grid[ni, gi]:.1f} dB",
                transform=ax.transAxes,
                fontsize=8, ha="center", va="top",
            )

    fig.suptitle(
        r"Effect of $\gamma$ across Noise Levels  (PDDR, $\ell_1$)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fname = _next_path("PDDR_GammaVarying", "png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {fname}")
    plt.show()


if __name__ == "__main__":
    main()
