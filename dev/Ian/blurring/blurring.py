def gaussian_psf(size=9, sigma=2.0):
    """
    Normalized Gaussian Blur Kernel
    """
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= np.sum(psf)
    return psf

"""Motion Blur"""
def motion_psf(length=9, angle=0.0):
    size = max(3, int(length))
    if size % 2 == 0:
        size += 1

    psf = np.zeros((size, size), dtype=np.float64)
    center = size // 2

    #Conevrt Units
    theta = np.deg2rad(angle)
    dx = np.cos(theta)
    dy = np.sin(theta)

    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, num=length):
        x = center + t * dx
        y = center + t * dy

        ix = int(round(x))
        iy = int(round(y))

        if 0 <= iy < size and 0 <= ix < size:
            psf[iy, ix] += 1.0

    s = psf.sum()
    if s == 0:
        psf[center, center] = 1.0
        s = 1.0

    psf /= s
    return psf

def build_kernel(kind="gaussian", size=9, sigma=2.0, motion_length=9, motion_angle=0.0):
    if kind == "gaussian":
        return gaussian_psf(size=size, sigma=sigma)
    elif kind == "motion":
        return motion_psf(length=motion_length, angle=motion_angle)
