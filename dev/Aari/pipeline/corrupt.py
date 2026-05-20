import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Blur kernels
# ---------------------------------------------------------------------------

def gaussian_psf(size=9, sigma=2.0):
    """
    Normalized 2-D Gaussian blur kernel.

    Parameters
    ----------
    size  : int   — kernel side length (odd recommended)
    sigma : float — standard deviation in pixels

    Returns
    -------
    psf : float64 ndarray, shape (size, size), sums to 1
    """
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return psf / psf.sum()


def motion_psf(length=9, angle=0.0):
    """
    Normalized motion-blur kernel.

    Parameters
    ----------
    length : int   — number of pixels the blur spans
    angle  : float — direction of motion in degrees (0 = horizontal)

    Returns
    -------
    psf : float64 ndarray, shape (length, length), sums to 1
    """
    psf = np.zeros((length, length), dtype=np.float64)
    center = length // 2
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    for i in range(length):
        offset = i - center
        col = int(round(center + offset * cos_a))
        row = int(round(center + offset * sin_a))
        if 0 <= row < length and 0 <= col < length:
            psf[row, col] += 1.0

    total = psf.sum()
    if total == 0:
        psf[center, center] = 1.0
    else:
        psf /= total
    return psf


def build_kernel(kind="gaussian", size=9, sigma=2.0,
                 motion_length=9, motion_angle=0.0):
    """
    Blur kernel factory.

    Parameters
    ----------
    kind         : str   — 'gaussian' or 'motion'
    size         : int   — kernel side length for Gaussian
    sigma        : float — std dev for Gaussian
    motion_length: int   — length for motion blur
    motion_angle : float — angle in degrees for motion blur

    Returns
    -------
    psf : float64 ndarray
    """
    if kind == "gaussian":
        return gaussian_psf(size=size, sigma=sigma)
    elif kind == "motion":
        return motion_psf(length=motion_length, angle=motion_angle)
    else:
        raise ValueError(f"Unknown blur kind '{kind}'. Choose 'gaussian' or 'motion'.")


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def add_noise(img, noise_type="gaussian", noise_level=0.01, seed=0):
    """
    Add noise to an image and clip to [0, 1].

    Parameters
    ----------
    img        : float64 ndarray, values in [0, 1]
    noise_type : str   — 'gaussian' or 'saltpepper'
    noise_level: float
        gaussian:    std dev of additive white Gaussian noise
        saltpepper:  fraction of pixels to corrupt (half → 0, half → 1)
    seed       : int   — RNG seed for reproducibility

    Returns
    -------
    noisy : float64 ndarray, same shape as img, values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    noisy = img.copy()

    if noise_type == "gaussian":
        noisy = noisy + noise_level * rng.standard_normal(img.shape)
    elif noise_type == "saltpepper":
        n_corrupt = int(noise_level * img.size)
        flat = noisy.ravel()
        indices = rng.choice(img.size, size=n_corrupt, replace=False)
        # first half → 0 (pepper), second half → 1 (salt)
        half = n_corrupt // 2
        flat[indices[:half]] = 0.0
        flat[indices[half:]] = 1.0
        noisy = flat.reshape(img.shape)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'. Choose 'gaussian' or 'saltpepper'.")

    return np.clip(noisy, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FFT helpers (circular convolution)
# ---------------------------------------------------------------------------

def _psf_to_otf(psf, shape):
    """
    Zero-pad PSF to `shape` and return its 2-D FFT.

    The PSF is centred so that the DC component is at (0, 0) after padding,
    giving circular (periodic) convolution semantics.

    Parameters
    ----------
    psf   : float64 ndarray, shape (kH, kW)
    shape : tuple (H, W) — target image shape

    Returns
    -------
    k_hat : complex128 ndarray, shape (H, W)
    """
    kH, kW = psf.shape
    H, W = shape

    padded = np.zeros((H, W), dtype=np.float64)
    padded[:kH, :kW] = psf

    # roll so the PSF centre lands at (0, 0)
    padded = np.roll(padded, shift=-(kH // 2), axis=0)
    padded = np.roll(padded, shift=-(kW // 2), axis=1)

    return np.fft.fft2(padded)


def _apply_blur(x, k_hat):
    """
    Apply circular convolution blur in the frequency domain.

    Parameters
    ----------
    x     : float64 ndarray, shape (H, W)
    k_hat : complex128 ndarray, shape (H, W) — output of _psf_to_otf

    Returns
    -------
    blurred : float64 ndarray, shape (H, W)
    """
    return np.real(np.fft.ifft2(k_hat * np.fft.fft2(x)))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_corrupted_image(
    image_path,
    blur_kind="gaussian",
    blur_ksize=9,
    blur_sigma=2.0,
    motion_length=9,
    motion_angle=0.0,
    noise_type="gaussian",
    noise_level=0.01,
    scale=1.0,
    seed=0,
):
    """
    Full image corruption pipeline: load → resize → blur → noise.

    Parameters
    ----------
    image_path   : str   — path to image file (JPEG, TIFF, PNG, …)
    blur_kind    : str   — 'gaussian' or 'motion'
    blur_ksize   : int   — kernel size for Gaussian blur
    blur_sigma   : float — std dev for Gaussian blur
    motion_length: int   — length for motion blur
    motion_angle : float — angle in degrees for motion blur
    noise_type   : str   — 'gaussian' or 'saltpepper'
    noise_level  : float — noise strength (see add_noise)
    scale        : float — resize factor applied before corruption (1.0 = no resize)
    seed         : int   — RNG seed for noise

    Returns
    -------
    x_true : float64 ndarray, shape (H, W), values in [0, 1] — clean image
    b      : float64 ndarray, shape (H, W), values in [0, 1] — blurred + noisy
    psf    : float64 ndarray, shape (ksize, ksize)           — spatial blur kernel
    """
    # 1. Load and convert to grayscale float64 in [0, 1]
    img = Image.open(image_path).convert("L")

    # 2. Optional resize
    if scale != 1.0:
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    x_true = np.asarray(img, dtype=np.float64) / 255.0

    # 3. Build blur kernel
    psf = build_kernel(
        kind=blur_kind,
        size=blur_ksize,
        sigma=blur_sigma,
        motion_length=motion_length,
        motion_angle=motion_angle,
    )

    # 4. Blur in frequency domain
    k_hat = _psf_to_otf(psf, x_true.shape)
    blurred = _apply_blur(x_true, k_hat)

    # 5. Add noise
    b = add_noise(blurred, noise_type=noise_type, noise_level=noise_level, seed=seed)

    return x_true, b, psf
