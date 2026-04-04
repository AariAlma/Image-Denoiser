"""
Combined Holistic Quality Metric
"""
def image_quality_metric(
    x_true,
    x_rec,
    data_range=1.0,
    sigma=1.5,
    patch_size=8,
    stride=4,
    reg=5e-2,
    maxiter=300,
    tol=1e-9,
    eps=1e-12,
    lambda_mass=1.0,
    lambda_l1=0.25,
    mass_weighting=True,
    tau_psnr=20.0,
    tau_disc=1.0,
    weights=(0.30, 0.35, 0.35),
):
    """
    Composite image quality metric on [0,100].

    Higher = better quality.

    This is being scored collectively on
      - PSNR score
      - SSIM score
      - intensity-aware local transport score
    """
    x_true = np.asarray(x_true, dtype=np.float64)
    x_rec = np.asarray(x_rec, dtype=np.float64)

    if x_true.shape != x_rec.shape:
        raise ValueError("x_true and x_rec must have the same shape")

    x_true = np.clip(x_true, 0.0, data_range)
    x_rec = np.clip(x_rec, 0.0, data_range)

    wP, wS, wT = weights
    wsum = wP + wS + wT
    if wsum <= 0:
        raise ValueError("weights must sum to a positive value")
    wP, wS, wT = wP / wsum, wS / wsum, wT / wsum

    psnr_val = psnr(x_true, x_rec, data_range=data_range, eps=eps)
    ssim_val = ssim(x_true, x_rec, data_range=data_range, sigma=sigma, eps=eps)

    transport_disc, w2_shape = patchwise_transport_discrepancy(
        x_true=x_true,
        x_rec=x_rec,
        patch_size=patch_size,
        stride=stride,
        reg=reg,
        maxiter=maxiter,
        tol=tol,
        eps=eps,
        lambda_mass=lambda_mass,
        lambda_l1=lambda_l1,
        mass_weighting=mass_weighting,
    )

    # Normalize each into [0,1], higher is better
    psnr_score = 1.0 - np.exp(-psnr_val / max(tau_psnr, eps))
    ssim_score = np.clip((1.0 + ssim_val) / 2.0, 0.0, 1.0)
    transport_score = np.exp(-transport_disc / max(tau_disc, eps))

    quality_01 = wP * psnr_score + wS * ssim_score + wT * transport_score
    quality_100 = 100.0 * quality_01

    return {
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "transport_discrepancy": float(transport_disc),
        "w2_shape_mean": float(w2_shape),
        "psnr_score": float(psnr_score),
        "ssim_score": float(ssim_score),
        "transport_score": float(transport_score),
        "quality_score_0_1": float(quality_01),
        "quality_score_0_100": float(quality_100),
        "weights": {
            "psnr": float(wP),
            "ssim": float(wS),
            "transport": float(wT),
        },
    }

""" Specified Quality Labels"""
def quality_label(score):
    score = float(score)

    if score >= 90:
        return "Close to SOTA"
    elif score >= 75:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Satisfactory"
    else:
        return "poor"
