import numpy as np
from scipy.ndimage import gaussian_filter

def psnr(x_true, x_rec, data_range=1.0, eps=1e-12):
    mse = np.mean((x_true - x_rec) ** 2)
    mse = max(mse, eps)
    return 10 * np.log10((data_range ** 2) / mse)
