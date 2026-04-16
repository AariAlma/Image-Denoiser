"""
gamma_vs_dimension.py  —  Does optimal gamma depend on image resolution?
=========================================================================
Runs golden section search for the optimal gamma across:
  - 4 image sizes (250px, 500px, 960px, 1920px wide — same scene)
  - 5 noise levels
  - 2 noise types  (Gaussian, Salt & Pepper)
  - 2 problem types (L1, L2)

The goal is to determine whether the optimal regularisation strength gamma
is scale-invariant (independent of image size/resolution) or whether it
needs to be re-tuned as resolution changes.

Run from the project/Aari/ directory:
    python gamma_vs_dimension.py

Outputs (written to project/Aari/):
    gamma_vs_dim_results.csv          — full numeric results table
    gamma_vs_dim_heatmap_<noise>.png  — gamma* + PSNR heatmap (size × noise level)
    gamma_vs_dim_lines_<noise>.png    — gamma* vs noise, one line per image size
    gamma_vs_dim_summary.png          — combined 4-panel summary across problems
"""

import csv
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import from Aari's pipeline and solver (script must be run from project/Aari/)
from pipeline import build_corrupted_image
from solver   import primal_dual_dr_solve


# =============================================================================
#  Configuration — edit here
# =============================================================================

# 4 pre-scaled versions of the same scene (full-size omitted: impractical overnight)
IMAGES = [
    ("250px",   "../testimages/testimages/250px-(Parque_La_Carolina,_Quito_antique_plane).jpg"),
    ("500px",   "../testimages/testimages/500px-(Parque_La_Carolina,_Quito_antique_plane).jpg"),
    ("960px",   "../testimages/testimages/960px-(Parque_La_Carolina,_Quito_antique_plane).jpg"),
    ("1920px",  "../testimages/testimages/1920px-(Parque_La_Carolina,_Quito_antique_plane).jpg"),
    # ("full", "../testimages/testimages/(Parque_La_Carolina,_Quito_antique_plane).jpg"),
    # ^ uncomment to include full-res (~4000px), but expect several extra hours
]

BLUR_KIND     = "gaussian"
BLUR_KSIZE    = 9
BLUR_SIGMA    = 2.0
SEED          = 0

NOISE_TYPES   = ["gaussian", "saltpepper"]
PROBLEMS      = ["l2", "l1"]       # L2 and L1; Huber excluded (has extra delta param)

# Per-problem solver parameters
SOLVER_PARAMS = {
    "l2": {"t": 1.0,  "rho": 1.0},
    "l1": {"t": 0.25, "rho": 0.5},
}

MAXITER       = 300        # reduced from 500 — still converges; saves ~40% runtime
TOL           = 1e-4
USE_GPU       = True       # set False to force CPU

NOISE_LEVELS  = [0.01, 0.05, 0.10, 0.20, 0.40]

# Golden section search bounds and tolerance (log-space)
LOG_GAMMA_MIN = math.log(1e-4)
LOG_GAMMA_MAX = math.log(2.0)
SEARCH_TOL    = 0.07       # slightly coarser than the 0.05 in gamma_search.py

# =============================================================================

_PHI = (math.sqrt(5) - 1) / 2   # ≈ 0.618 — golden ratio conjugate

NOISE_LABELS = {"gaussian": "Gaussian", "saltpepper": "Salt & Pepper"}


# -----------------------------------------------------------------------------
#  PSNR helper
# -----------------------------------------------------------------------------

def _psnr(x_sol, x_true):
    mse = float(np.mean((x_sol - x_true) ** 2))
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")


# -----------------------------------------------------------------------------
#  Single solver evaluation
# -----------------------------------------------------------------------------

def _evaluate(log_gamma, b, psf, x_true, problem):
    sp = SOLVER_PARAMS[problem]
    x_sol, _ = primal_dual_dr_solve(
        b, psf,
        problem=problem,
        gamma=math.exp(log_gamma),
        t=sp["t"],
        rho=sp["rho"],
        maxiter=MAXITER,
        tol=TOL,
        verbose=False,
        use_gpu=USE_GPU,
    )
    return _psnr(x_sol, x_true)


# -----------------------------------------------------------------------------
#  Golden section search (maximise PSNR over log-gamma)
# -----------------------------------------------------------------------------

def golden_search(b, psf, x_true, problem):
    """
    Returns
    -------
    best_gamma : float
    best_psnr  : float
    history    : list of (gamma, psnr) — one entry per solver call
    """
    lo, hi = LOG_GAMMA_MIN, LOG_GAMMA_MAX
    c = hi - _PHI * (hi - lo)
    d = lo + _PHI * (hi - lo)

    fc = _evaluate(c, b, psf, x_true, problem)
    fd = _evaluate(d, b, psf, x_true, problem)
    history = [(math.exp(c), fc), (math.exp(d), fd)]

    step = 2
    print(f"        step  1  gamma={math.exp(c):.5f}  PSNR={fc:.3f} dB")
    print(f"        step  2  gamma={math.exp(d):.5f}  PSNR={fd:.3f} dB")

    while hi - lo > SEARCH_TOL:
        step += 1
        if fc > fd:
            hi = d;  d, fd = c, fc
            c  = hi - _PHI * (hi - lo)
            fc = _evaluate(c, b, psf, x_true, problem)
            history.append((math.exp(c), fc))
            print(f"        step {step:2d}  gamma={math.exp(c):.5f}  PSNR={fc:.3f} dB")
        else:
            lo = c;  c, fc = d, fd
            d  = lo + _PHI * (hi - lo)
            fd = _evaluate(d, b, psf, x_true, problem)
            history.append((math.exp(d), fd))
            print(f"        step {step:2d}  gamma={math.exp(d):.5f}  PSNR={fd:.3f} dB")

    best_gamma = math.exp((lo + hi) / 2)
    best_psnr  = max(fc, fd)
    return best_gamma, best_psnr, history


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main():
    size_labels = [s for s, _ in IMAGES]
    total = len(IMAGES) * len(NOISE_TYPES) * len(PROBLEMS) * len(NOISE_LEVELS)
    done  = 0
    t_start_all = time.time()

    # results: list of dicts, one per (size, noise_type, problem, noise_level)
    results = []

    for size_label, img_path in IMAGES:
        for noise_type in NOISE_TYPES:
            for problem in PROBLEMS:
                for noise_level in NOISE_LEVELS:
                    done += 1
                    elapsed_all = time.time() - t_start_all
                    eta_str = ""
                    if done > 1:
                        per_task = elapsed_all / (done - 1)
                        remaining = per_task * (total - done + 1)
                        eta_str = f"  ETA ~{remaining/3600:.1f}h"

                    print(f"\n[{done}/{total}{eta_str}]  "
                          f"size={size_label}  noise={noise_type}({noise_level:.2f})  "
                          f"problem={problem.upper()}")

                    t0 = time.time()
                    try:
                        x_true, b, psf = build_corrupted_image(
                            img_path,
                            blur_kind=BLUR_KIND,
                            blur_ksize=BLUR_KSIZE,
                            blur_sigma=BLUR_SIGMA,
                            noise_type=noise_type,
                            noise_level=noise_level,
                            seed=SEED,
                        )
                        H, W = x_true.shape
                        n_pixels = H * W
                        print(f"    image: {H}x{W} = {n_pixels:,} pixels")

                        best_gamma, best_psnr, history = golden_search(
                            b, psf, x_true, problem
                        )
                        dt = time.time() - t0
                        print(f"    --> gamma*={best_gamma:.5f}  "
                              f"PSNR={best_psnr:.3f} dB  "
                              f"evals={len(history)}  "
                              f"time={dt:.1f}s")

                        results.append({
                            "size_label":    size_label,
                            "n_pixels":      n_pixels,
                            "height":        H,
                            "width":         W,
                            "noise_type":    noise_type,
                            "noise_level":   noise_level,
                            "problem":       problem,
                            "optimal_gamma": round(best_gamma, 6),
                            "best_psnr":     round(best_psnr,  4),
                            "n_evals":       len(history),
                            "time_sec":      round(dt, 1),
                        })

                    except Exception as exc:
                        print(f"    ERROR: {exc}")
                        results.append({
                            "size_label":    size_label,
                            "n_pixels":      None,
                            "height":        None,
                            "width":         None,
                            "noise_type":    noise_type,
                            "noise_level":   noise_level,
                            "problem":       problem,
                            "optimal_gamma": float("nan"),
                            "best_psnr":     float("nan"),
                            "n_evals":       0,
                            "time_sec":      round(time.time() - t0, 1),
                        })

    total_time = time.time() - t_start_all
    print(f"\n{'='*60}")
    print(f"  All done in {total_time/3600:.2f} hours")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    #  Save CSV
    # -------------------------------------------------------------------------
    csv_fname = "gamma_vs_dim_results.csv"
    with open(csv_fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved --> {csv_fname}")

    # -------------------------------------------------------------------------
    #  Build look-up: npix_map[size_label] = n_pixels
    # -------------------------------------------------------------------------
    npix_map = {}
    for r in results:
        if r["size_label"] not in npix_map and r["n_pixels"] is not None:
            npix_map[r["size_label"]] = r["n_pixels"]

    size_colors = plt.cm.tab10(np.linspace(0, 0.5, len(size_labels)))

    # =========================================================================
    #  Per-noise-type figures
    # =========================================================================
    for noise_type in NOISE_TYPES:
        nt_label = NOISE_LABELS[noise_type]

        # Build matrices:  shape = (n_sizes, n_levels) per problem
        gamma_mats = {}
        psnr_mats  = {}
        for problem in PROBLEMS:
            gmat = np.full((len(size_labels), len(NOISE_LEVELS)), np.nan)
            pmat = np.full((len(size_labels), len(NOISE_LEVELS)), np.nan)
            for r in results:
                if r["noise_type"] != noise_type or r["problem"] != problem:
                    continue
                si = size_labels.index(r["size_label"])
                li = NOISE_LEVELS.index(r["noise_level"])
                gmat[si, li] = r["optimal_gamma"]
                pmat[si, li] = r["best_psnr"]
            gamma_mats[problem] = gmat
            psnr_mats[problem]  = pmat

        # ---- Figure 1: heatmap of log10(gamma*) for each problem ------------
        fig, axes = plt.subplots(
            len(PROBLEMS), 2,
            figsize=(12, 4 * len(PROBLEMS)),
        )
        if len(PROBLEMS) == 1:
            axes = axes[np.newaxis, :]   # ensure 2-D indexing

        y_tick_labels = [
            f"{sl}  ({npix_map.get(sl, '?'):,} px)"
            for sl in size_labels
        ]

        for ri, problem in enumerate(PROBLEMS):
            gmat = np.log10(np.clip(gamma_mats[problem], 1e-8, None))
            pmat = psnr_mats[problem]

            for ci, (mat, title, cmap, fmt) in enumerate([
                (gmat, f"log10(gamma*)  [{problem.upper()}]", "viridis",  ".2f"),
                (pmat, f"Best PSNR (dB) [{problem.upper()}]", "plasma",   ".1f"),
            ]):
                ax = axes[ri, ci]
                vmin, vmax = np.nanmin(mat), np.nanmax(mat)
                im = ax.imshow(mat, aspect="auto", origin="upper",
                               cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks(range(len(NOISE_LEVELS)))
                ax.set_xticklabels([f"{l:.2f}" for l in NOISE_LEVELS])
                ax.set_yticks(range(len(size_labels)))
                ax.set_yticklabels(y_tick_labels)
                ax.set_xlabel("Noise level", fontsize=9)
                ax.set_ylabel("Image size", fontsize=9)
                ax.set_title(title, fontsize=10)
                plt.colorbar(im, ax=ax)
                # annotate cells
                for i in range(len(size_labels)):
                    for j in range(len(NOISE_LEVELS)):
                        v = mat[i, j]
                        if not np.isnan(v):
                            mid = (vmin + vmax) / 2
                            c_txt = "white" if v < mid else "black"
                            ax.text(j, i, f"{v:{fmt}}", ha="center",
                                    va="center", fontsize=8, color=c_txt)

        fig.suptitle(
            f"Optimal gamma* and PSNR vs Image Size x Noise Level\n"
            f"({nt_label} noise, PDDR solver, Gaussian blur 9x9 sigma=2)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        fname = f"gamma_vs_dim_heatmap_{noise_type}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        print(f"Saved --> {fname}")
        plt.close(fig)

        # ---- Figure 2: gamma* vs noise level, lines per image size ----------
        fig2, axes2 = plt.subplots(
            1, len(PROBLEMS),
            figsize=(7 * len(PROBLEMS), 5),
            sharey=False,
        )
        if len(PROBLEMS) == 1:
            axes2 = [axes2]

        for ci, problem in enumerate(PROBLEMS):
            ax = axes2[ci]
            gmat = gamma_mats[problem]
            for si, sl in enumerate(size_labels):
                npix = npix_map.get(sl, 0)
                lbl  = f"{sl}  ({npix:,} px)"
                ax.plot(
                    NOISE_LEVELS, gmat[si, :],
                    marker="o", linewidth=1.8, markersize=6,
                    color=size_colors[si], label=lbl,
                )
            ax.set_yscale("log")
            ax.set_xlabel("Noise level", fontsize=10)
            ax.set_ylabel("Optimal gamma*  (log scale)", fontsize=10)
            ax.set_title(f"{problem.upper()} problem", fontsize=11)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

        fig2.suptitle(
            f"Optimal gamma* vs Noise Level by Image Size\n"
            f"({nt_label} noise, PDDR, Gaussian blur 9x9 sigma=2)",
            fontsize=12, fontweight="bold",
        )
        fig2.tight_layout()
        fname2 = f"gamma_vs_dim_lines_{noise_type}.png"
        fig2.savefig(fname2, dpi=130, bbox_inches="tight")
        print(f"Saved --> {fname2}")
        plt.close(fig2)

        # ---- Figure 3: gamma* vs n_pixels (key question: scale invariance?) -
        fig3, axes3 = plt.subplots(
            1, len(PROBLEMS),
            figsize=(7 * len(PROBLEMS), 5),
            sharey=False,
        )
        if len(PROBLEMS) == 1:
            axes3 = [axes3]

        level_colors = plt.cm.coolwarm(np.linspace(0, 1, len(NOISE_LEVELS)))

        for ci, problem in enumerate(PROBLEMS):
            ax = axes3[ci]
            gmat = gamma_mats[problem]
            n_pixels_list = [npix_map.get(sl, np.nan) for sl in size_labels]

            for li, nl in enumerate(NOISE_LEVELS):
                gammas = gmat[:, li]
                ax.plot(
                    n_pixels_list, gammas,
                    marker="s", linewidth=1.8, markersize=6,
                    color=level_colors[li],
                    label=f"noise={nl:.2f}",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Image size (total pixels, log scale)", fontsize=10)
            ax.set_ylabel("Optimal gamma*  (log scale)", fontsize=10)
            ax.set_title(f"{problem.upper()} — gamma* vs image size", fontsize=11)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

        fig3.suptitle(
            f"Scale Invariance Check: gamma* vs Total Pixels\n"
            f"({nt_label} noise, PDDR, Gaussian blur 9x9 sigma=2)\n"
            f"Flat lines = gamma* is scale-invariant; "
            f"Rising/falling = needs rescaling",
            fontsize=11, fontweight="bold",
        )
        fig3.tight_layout()
        fname3 = f"gamma_vs_dim_scaleinv_{noise_type}.png"
        fig3.savefig(fname3, dpi=130, bbox_inches="tight")
        print(f"Saved --> {fname3}")
        plt.close(fig3)

    # =========================================================================
    #  Summary figure: all 4 combos (2 noise × 2 problems) on one page
    # =========================================================================
    fig4, axes4 = plt.subplots(
        len(NOISE_TYPES), len(PROBLEMS),
        figsize=(7 * len(PROBLEMS), 5 * len(NOISE_TYPES)),
        sharey=False,
    )

    for ri, noise_type in enumerate(NOISE_TYPES):
        for ci, problem in enumerate(PROBLEMS):
            ax = axes4[ri, ci]
            gmat = gamma_mats if noise_type == NOISE_TYPES[-1] else None
            # rebuild for this specific combo
            gmat_here = np.full((len(size_labels), len(NOISE_LEVELS)), np.nan)
            for r in results:
                if r["noise_type"] != noise_type or r["problem"] != problem:
                    continue
                si = size_labels.index(r["size_label"])
                li = NOISE_LEVELS.index(r["noise_level"])
                gmat_here[si, li] = r["optimal_gamma"]

            for si, sl in enumerate(size_labels):
                npix = npix_map.get(sl, 0)
                ax.plot(
                    NOISE_LEVELS, gmat_here[si, :],
                    marker="o", linewidth=1.8, markersize=5,
                    color=size_colors[si],
                    label=f"{sl} ({npix:,} px)",
                )
            ax.set_yscale("log")
            ax.set_xlabel("Noise level", fontsize=9)
            ax.set_ylabel("gamma*  (log scale)", fontsize=9)
            ax.set_title(
                f"{NOISE_LABELS[noise_type]} / {problem.upper()}",
                fontsize=10,
            )
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, which="both", linestyle="--", alpha=0.35)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    fig4.suptitle(
        "Optimal gamma* vs Noise Level by Image Size — All Combinations\n"
        "(PDDR, Gaussian blur 9x9 sigma=2.0)",
        fontsize=13, fontweight="bold",
    )
    fig4.tight_layout()
    fname4 = "gamma_vs_dim_summary.png"
    fig4.savefig(fname4, dpi=130, bbox_inches="tight")
    print(f"Saved --> {fname4}")
    plt.close(fig4)

    # =========================================================================
    #  Print compact results table
    # =========================================================================
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    header = f"{'Size':>8}  {'Noise type':>12}  {'Problem':>7}  "
    for nl in NOISE_LEVELS:
        header += f"  nl={nl:.2f}"
    print(header)
    print("-" * len(header))
    for size_label in size_labels:
        for noise_type in NOISE_TYPES:
            for problem in PROBLEMS:
                row = f"{size_label:>8}  {NOISE_LABELS[noise_type]:>12}  {problem.upper():>7}  "
                for nl in NOISE_LEVELS:
                    hit = next(
                        (r for r in results
                         if r["size_label"] == size_label
                         and r["noise_type"] == noise_type
                         and r["problem"] == problem
                         and abs(r["noise_level"] - nl) < 1e-9),
                        None,
                    )
                    val = hit["optimal_gamma"] if hit else float("nan")
                    row += f"  {val:7.4f}"
                print(row)

    print(f"\nTotal time: {total_time/3600:.2f} hours")
    print("All outputs saved in project/Aari/")


if __name__ == "__main__":
    main()
