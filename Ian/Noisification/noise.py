""" Noise Model
    noise_type : str
        'gaussian' or 'saltpepper'
    noise_level : float
        For gaussian: standard deviation
        For saltpepper: fraction of corrupted pixels
    seed : int
        Random seed.
    Notw - Clipped output
    """
def add_noise(img, noise_type="gaussian", noise_level=0.01, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()

    if noise_type == "gaussian":
        out = out + noise_level * rng.standard_normal(img.shape)

    elif noise_type == "saltpepper":
        mask = rng.random(img.shape)
        out[mask < noise_level / 2] = 0.0
        out[(mask >= noise_level / 2) & (mask < noise_level)] = 1.0

    else:
        raise ValueError("noise_type must be 'gaussian' or 'saltpepper'")

    return np.clip(out, 0.0, 1.0)
