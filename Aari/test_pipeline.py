import matplotlib.pyplot as plt
from pipeline import build_corrupted_image

x, b, psf = build_corrupted_image(
    "../testimages/testimages/cameraman.jpg",
    blur_kind="gaussian", blur_ksize=9, blur_sigma=2.0,
    noise_type="gaussian", noise_level=0.02,
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(x,   cmap="gray"); axes[0].set_title("Original");       axes[0].axis("off")
axes[1].imshow(b,   cmap="gray"); axes[1].set_title("Blurred + Noisy"); axes[1].axis("off")
axes[2].imshow(psf, cmap="hot");  axes[2].set_title("PSF kernel");      axes[2].axis("off")
plt.tight_layout()
plt.show()
