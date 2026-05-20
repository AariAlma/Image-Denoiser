def ssim(x_true, x_rec, data_range=1.0, sigma=1.5, eps=1e-12):
    x_true = x_true.astype(np.float64)
    x_rec = x_rec.astype(np.float64)
    mu_x = gaussian_filter(x_true, sigma)
    mu_y = gaussian_filter(x_rec, sigma) # Gaussian filtering to get local means
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variances and covariance
    sigma_x2 = gaussian_filter(x_true ** 2, sigma) - mu_x2
    sigma_y2 = gaussian_filter(x_rec ** 2, sigma) - mu_y2
    
    sigma_xy = gaussian_filter(x_true * x_rec, sigma) - mu_xy
    # Stability constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = numerator / (denominator + eps)

    return np.mean(ssim_map)
