"metric_sigma": 1.5,
        "metric_patch_size": 8,
        "metric_stride": 4,
        "metric_reg": 5e-2,
        "metric_maxiter": 300,
        "metric_tol": 1e-9,
        "metric_lambda_mass": 1.0,
        "metric_lambda_l1": 0.25,
        "metric_mass_weighting": True,
        "metric_tau_psnr": 20.0,
        "metric_tau_disc": 1.0,
        "metric_weights": (0.25, 0.35, 0.40),

metric = image_quality_metric(
        x_true=x_true,
        x_rec=x_rec,
        data_range=1.0,
        sigma=config["metric_sigma"],
        patch_size=config["metric_patch_size"],
        stride=config["metric_stride"],
        reg=config["metric_reg"],
        maxiter=config["metric_maxiter"],
        tol=config["metric_tol"],
        lambda_mass=config["metric_lambda_mass"],
        lambda_l1=config["metric_lambda_l1"],
        mass_weighting=config["metric_mass_weighting"],
        tau_psnr=config["metric_tau_psnr"],
        tau_disc=config["metric_tau_disc"],
        weights=config["metric_weights"],
    )

 print("Raw Quality Metrics")
    print(f"PSNR                      : {metric['psnr']:.6f}")
    print(f"SSIM                      : {metric['ssim']:.6f}")
    print(f"Transport discrepancy     : {metric['transport_discrepancy']:.6f}")
    print(f"Mean local W2-shape       : {metric['w2_shape_mean']:.6f}")

    print("Normalized Quality Metrics")
    print(f"PSNR score                : {metric['psnr_score']:.6f}")
    print(f"SSIM score                : {metric['ssim_score']:.6f}")
    print(f"Transport score           : {metric['transport_score']:.6f}")

    print("Composite Quality Metric")
    print(f"Quality score [0,1]       : {metric['quality_score_0_1']:.6f}")
    print(f"Quality score [0,100]     : {metric['quality_score_0_100']:.6f}")
    print(f"Quality label             : {quality_label(metric['quality_score_0_100'])}")
