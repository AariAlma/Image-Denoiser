import matplotlib.pyplot as plt
from pipeline import build_corrupted_image
from solver import primal_dual_dr_solve

# ---- Configuration --------------------------------------------------------
config = {
    # Image
    "image":         "../testimages/testimages/cameraman.jpg",
    "scale":         1.0,

    # Blur
    "blur_kind":     "gaussian",   # "gaussian" | "motion"
    "blur_ksize":    9,
    "blur_sigma":    2.0,
    "motion_length": 9,
    "motion_angle":  0.0,

    # Noise
    "noise_type":    "gaussian",   # "gaussian" | "saltpepper"
    "noise_level":   0.02,
    "seed":          0,

    # Solver
    "problem":       "l2",         # "l1" | "l2"
    "gamma":         0.006,
    "t":             1.0,
    "rho":           1.0,
    "maxiter":       500,
    "tol":           1e-4,
}
# ---------------------------------------------------------------------------

x_true, b, psf = build_corrupted_image(
    config["image"],
    blur_kind=config["blur_kind"],
    blur_ksize=config["blur_ksize"],
    blur_sigma=config["blur_sigma"],
    motion_length=config["motion_length"],
    motion_angle=config["motion_angle"],
    noise_type=config["noise_type"],
    noise_level=config["noise_level"],
    scale=config["scale"],
    seed=config["seed"],
)

print(f"Image: {x_true.shape}  problem={config['problem']}  gamma={config['gamma']}")

x_sol, info = primal_dual_dr_solve(
    b, psf,
    problem=config["problem"],
    gamma=config["gamma"],
    t=config["t"],
    rho=config["rho"],
    maxiter=config["maxiter"],
    tol=config["tol"],
    verbose=True,
    print_every=50,
)

# ---- Display ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(x_true, cmap="gray"); axes[0].set_title("Original");         axes[0].axis("off")
axes[1].imshow(b,      cmap="gray"); axes[1].set_title("Blurred + Noisy");   axes[1].axis("off")
axes[2].imshow(x_sol,  cmap="gray"); axes[2].set_title(f"PDDR ({config['problem'].upper()})"); axes[2].axis("off")
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.semilogy(info["objective"])
ax2.set_xlabel("Iteration"); ax2.set_ylabel("Objective"); ax2.set_title("Convergence")
plt.tight_layout()
plt.show()
