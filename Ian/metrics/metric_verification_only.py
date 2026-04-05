import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

CONFIG = {
    "image_1": "cameraman.jpg",      # first image
    "image_2": "reconstruction.png",   # second image

    # This is for resizing if needed
    # "match_1_to_2"  -> resize image_1 to image_2 size
    # "match_2_to_1"  -> resize image_2 to image_1 size
    "resize_mode": "match_2_to_1",

    # Metric parameters
    "patch_size": 3,
    "stride": 4,
    "reg": 5e-2,
    "sinkhorn_maxiter": 300,
    "sinkhorn_tol": 1e-9,
    "lambda_mass": 1.0,
    "lambda_l1": 0.25,
    "tau_psnr": 20.0,
    "tau_disc": 1.0,
    "weights": (0.30, 0.35, 0.35),
}


# Image Utilities
def load_grayscale_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found - {path}")

    img = Image.open(path).convert("L")
    img = np.asarray(img, dtype=np.float64)

    img -= img.min()
    max_val = img.max()
    if max_val > 0:
        img /= max_val

    return img


def resize_to_match(img, target_shape):
    """
    Resize grayscale image to target_shape = (H, W).
    """
    target_h, target_w = target_shape
    pil_img = Image.fromarray((255.0 * np.clip(img, 0.0, 1.0)).astype(np.uint8))
    pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    out = np.asarray(pil_img, dtype=np.float64) / 255.0
    return out




# Metrics
def psnr_metric(x_true, x_rec, data_range=1.0, eps=1e-12):
    x_true = np.asarray(x_true, dtype=np.float64)
    x_rec = np.asarray(x_rec, dtype=np.float64)
    mse = np.mean((x_true - x_rec) ** 2)
    mse = max(mse, eps)
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim_metric(x_true, x_rec, data_range=1.0, sigma=1.5, eps=1e-12):
    x_true = np.asarray(x_true, dtype=np.float64)
    x_rec = np.asarray(x_rec, dtype=np.float64)

    mu_x = gaussian_filter(x_true, sigma=sigma)
    mu_y = gaussian_filter(x_rec, sigma=sigma)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_filter(x_true ** 2, sigma=sigma) - mu_x2
    sigma_y2 = gaussian_filter(x_rec ** 2, sigma=sigma) - mu_y2
    sigma_xy = gaussian_filter(x_true * x_rec, sigma=sigma) - mu_xy

    sigma_x2 = np.maximum(sigma_x2, 0.0)
    sigma_y2 = np.maximum(sigma_y2, 0.0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = numerator / (denominator + eps)
    return float(np.mean(ssim_map))



# Local Transport Mismatch
def _normalize_patch_to_mass(patch, eps=1e-12):
    patch = np.asarray(patch, dtype=np.float64)
    patch = np.clip(patch, 0.0, None)

    total = patch.sum()
    if total <= eps:
        return None, 0.0

    return (patch / total).ravel(), float(total)


def _sinkhorn_w2_squared(a, b, C, reg=5e-2, maxiter=300, tol=1e-9, eps=1e-12):
    K = np.exp(-C / max(reg, eps))
    K = np.maximum(K, eps)

    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(maxiter):
        u_prev = u.copy()

        Kv = K @ v
        Kv = np.maximum(Kv, eps)
        u = a / Kv

        KTu = K.T @ u
        KTu = np.maximum(KTu, eps)
        v = b / KTu

        if np.linalg.norm(u - u_prev, ord=1) < tol:
            break

    P = (u[:, None] * K) * v[None, :]
    return float(np.sum(P * C))


def patchwise_transport_discrepancy(
    x_true,
    x_rec,
    patch_size=8,
    stride=4,
    reg=5e-2,
    maxiter=300,
    tol=1e-9,
    eps=1e-12,
    lambda_mass=1.0,
    lambda_l1=0.25,
    mass_weighting=True,
):
    x_true = np.asarray(x_true, dtype=np.float64)
    x_rec = np.asarray(x_rec, dtype=np.float64)

    if x_true.ndim != 2 or x_rec.ndim != 2:
        raise ValueError("Expected two grayscale 2D images")
    if x_true.shape != x_rec.shape:
        raise ValueError("Images must have the same shape")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")

    h, w = x_true.shape
    if h < patch_size or w < patch_size:
        raise ValueError("patch_size is larger than the image dimensions.")

    rows, cols = np.meshgrid(
        np.arange(patch_size), np.arange(patch_size), indexing="ij"
    )
    coords = np.column_stack([rows.ravel(), cols.ravel()])
    diff = coords[:, None, :] - coords[None, :, :]
    C = np.sum(diff ** 2, axis=2)
    max_cost = float(np.max(C))

    vals = []
    w2_vals = []
    weights = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            p_true = np.clip(x_true[i:i + patch_size, j:j + patch_size], 0.0, 1.0)
            p_rec = np.clip(x_rec[i:i + patch_size, j:j + patch_size], 0.0, 1.0)

            l1_local = float(np.mean(np.abs(p_true - p_rec)))

            a, m_true = _normalize_patch_to_mass(p_true, eps=eps)
            b, m_rec = _normalize_patch_to_mass(p_rec, eps=eps)

            if a is None and b is None:
                ot_val = 0.0
                mass_mismatch = 0.0
                weight = 1.0
            else:
                if a is None or b is None:
                    ot_val = max_cost
                else:
                    ot_val = _sinkhorn_w2_squared(
                        a=a, b=b, C=C, reg=reg, maxiter=maxiter, tol=tol, eps=eps
                    )

                mass_den = max(m_true, m_rec, eps)
                mass_mismatch = abs(m_true - m_rec) / mass_den
                weight = 0.5 * (m_true + m_rec) if mass_weighting else 1.0

            local_val = ot_val + lambda_mass * mass_mismatch + lambda_l1 * l1_local

            vals.append(local_val)
            w2_vals.append(np.sqrt(max(ot_val, 0.0)))
            weights.append(weight)

    vals = np.asarray(vals, dtype=np.float64)
    w2_vals = np.asarray(w2_vals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if np.all(weights <= eps):
        return 0.0, 0.0

    disc_mean = float(np.sum(weights * vals) / np.sum(weights))
    w2_shape_mean = float(np.sum(weights * w2_vals) / np.sum(weights))
    return disc_mean, w2_shape_mean



# Combined 
def image_quality_metric(
    x_true,
    x_rec,
    data_range=1.0,
    patch_size=8,
    stride=4,
    reg=5e-2,
    metric_maxiter=300,
    metric_tol=1e-9,
    lambda_mass=1.0,
    lambda_l1=0.25,
    tau_psnr=20.0,
    tau_disc=1.0,
    weights=(0.30, 0.35, 0.35),
):
    """
    This will produce a 
    - quality score - higher is better, and a 
    - mismatch score - lower is better
    """
    x_true = np.clip(np.asarray(x_true, dtype=np.float64), 0.0, data_range)
    x_rec = np.clip(np.asarray(x_rec, dtype=np.float64), 0.0, data_range)

    if x_true.shape != x_rec.shape:
        raise ValueError("Images must have the same shpae before metric computation")

    wP, wS, wT = weights
    wsum = wP + wS + wT
    if wsum <= 0:
        raise ValueError("weights must sum to a positive value")
    wP, wS, wT = wP / wsum, wS / wsum, wT / wsum

    psnr_val = psnr_metric(x_true, x_rec, data_range=data_range)
    ssim_val = ssim_metric(x_true, x_rec, data_range=data_range)

    transport_disc, w2_shape = patchwise_transport_discrepancy(
        x_true=x_true,
        x_rec=x_rec,
        patch_size=patch_size,
        stride=stride,
        reg=reg,
        maxiter=metric_maxiter,
        tol=metric_tol,
        lambda_mass=lambda_mass,
        lambda_l1=lambda_l1,
    )

    psnr_score = 1.0 - np.exp(-psnr_val / max(tau_psnr, 1e-12))
    ssim_score = np.clip((1.0 + ssim_val) / 2.0, 0.0, 1.0)
    transport_score = np.exp(-transport_disc / max(tau_disc, 1e-12))

    quality_01 = wP * psnr_score + wS * ssim_score + wT * transport_score
    quality_100 = 100.0 * quality_01
    mismatch_100 = 100.0 - quality_100

    return {
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "transport_discrepancy": float(transport_disc),
        "w2_shape_mean": float(w2_shape),
        "psnr_score": float(psnr_score),
        "ssim_score": float(ssim_score),
        "transport_score": float(transport_score),
        "quality_score_0_100": float(quality_100),
        "mismatch_score_0_100": float(mismatch_100),
    }


def quality_label(score):
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "fair"
    if score >= 40:
        return "poor"
    return "very poor"


def mismatch_label(score):
    if score <= 15:
        return "very small mismatch"
    if score <= 25:
        return "small mismatch"
    if score <= 40:
        return "moderate mismatch"
    if score <= 55:
        return "large mismatch"
    return "very large mismatch"


"""
Display - 
This displays both the images side-by-side first and then the information-mismatch (colored, if too bright or illegible, then change colour)
"""
def show_two_images(img1, img2, title1="Image 1", title2="Image 2"):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap="gray")
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap="gray")
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_absolute_difference(img1, img2):
    diff = np.abs(img1 - img2)

    plt.figure(figsize=(5, 4))
    plt.imshow(diff, cmap="hot")
    plt.title("Absolute Difference")
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.show()




"""Main"""
def main():
    cfg = CONFIG

    img1 = load_grayscale_image(cfg["image_1"])
    img2 = load_grayscale_image(cfg["image_2"])

    if img1.shape != img2.shape:
        if cfg["resize_mode"] == "match_1_to_2":
            img1 = resize_to_match(img1, img2.shape)
        elif cfg["resize_mode"] == "match_2_to_1":
            img2 = resize_to_match(img2, img1.shape)
        else:
            raise ValueError("resize_mode must be 'match_1_to_2' or 'match_2_to_1'")

    metric = image_quality_metric(
        x_true=img1,
        x_rec=img2,
        patch_size=cfg["patch_size"],
        stride=cfg["stride"],
        reg=cfg["reg"],
        metric_maxiter=cfg["sinkhorn_maxiter"],
        metric_tol=cfg["sinkhorn_tol"],
        lambda_mass=cfg["lambda_mass"],
        lambda_l1=cfg["lambda_l1"],
        tau_psnr=cfg["tau_psnr"],
        tau_disc=cfg["tau_disc"],
        weights=cfg["weights"],
    )

    print("\n Raw Comparison")
    print(f"PSNR                      : {metric['psnr']:.6f}")
    print(f"SSIM                      : {metric['ssim']:.6f}")
    print(f"Transport discrepancy     : {metric['transport_discrepancy']:.6f}")
    print(f"Mean local W2-shape       : {metric['w2_shape_mean']:.6f}")

    print("\n Normalized Comparison Scores")
    print(f"PSNR score                : {metric['psnr_score']:.6f}")
    print(f"SSIM score                : {metric['ssim_score']:.6f}")
    print(f"Transport score           : {metric['transport_score']:.6f}")

    print("\n Final")
    print(f"Quality score [0,100]     : {metric['quality_score_0_100']:.6f}")
    print(f"Mismatch score [0,100]    : {metric['mismatch_score_0_100']:.6f}")
    print(f"Quality label             : {quality_label(metric['quality_score_0_100'])}")
    print(f"Mismatch label            : {mismatch_label(metric['mismatch_score_0_100'])}")

    show_two_images(img1, img2, title1="Image 1", title2="Image 2")
    show_absolute_difference(img1, img2)


if __name__ == "__main__":
    main()
