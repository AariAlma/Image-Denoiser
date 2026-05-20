import numpy as np
import matplotlib.pyplot as plt
from pipeline import build_corrupted_image
from solver import primal_dual_dr_solve

# ---- Setup ----------------------------------------------------------------
x_true, b, psf = build_corrupted_image(
    "../testimages/testimages/cameraman.jpg",
    blur_kind="gaussian", blur_ksize=9, blur_sigma=2.0,
    noise_type="gaussian", noise_level=0.02,
)

# ---- Run solver -----------------------------------------------------------
x_sol, info = primal_dual_dr_solve(
    b, psf, problem="l2", gamma=0.006,
    t=1.0, rho=1.0, maxiter=500, tol=1e-4,
    verbose=True, print_every=50,
)

# ---- Metrics --------------------------------------------------------------
mse_input = float(np.mean((b      - x_true)**2))
mse_sol   = float(np.mean((x_sol  - x_true)**2))
psnr = lambda mse: 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

print(f"\n  MSE  (blurred+noisy) : {mse_input:.6f}  |  PSNR: {psnr(mse_input):.2f} dB")
print(f"  MSE  (reconstructed) : {mse_sol:.6f}  |  PSNR: {psnr(mse_sol):.2f} dB")

# ---- Visual ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(x_true, cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Original")
axes[0].axis("off")

axes[1].imshow(b, cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Blurred + Noisy\nPSNR: {psnr(mse_input):.2f} dB")
axes[1].axis("off")

axes[2].imshow(x_sol, cmap="gray", vmin=0, vmax=1)
axes[2].set_title(f"PDDR Reconstruction\nPSNR: {psnr(mse_sol):.2f} dB")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# ---- Convergence ----------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3))
axes2[0].semilogy(info["objective"])
axes2[0].set_xlabel("Iteration"); axes2[0].set_ylabel("Objective value")
axes2[0].set_title("Objective")

axes2[1].semilogy(info["rel_change"])
axes2[1].set_xlabel("Iteration"); axes2[1].set_ylabel("Relative change")
axes2[1].set_title("Convergence")

plt.tight_layout()
plt.show()
