"""
Local Optimal Transport Distance
"""
def _normalize_patch_to_mass(patch, eps=1e-12):
    """
    Normalize a nonnegative patch into a probability vector
    Returns (probability_vector, total_mass).
    """
    patch = np.asarray(patch, dtype=np.float64)
    patch = np.clip(patch, 0.0, None)

    total = patch.sum()
    if total <= eps:
        return None, 0.0

    return (patch / total).ravel(), float(total)


# This is a simpler version of the full Wasserstein IPM
def _sinkhorn_w2_squared(a, b, C, reg=5e-2, maxiter=300, tol=1e-9, eps=1e-12):
    """
    Sinkhorn approximation for discrete quadratic-cost OT.
    Returns -  approximate squared W2 cost.
    """
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
    """

        local discrepancy = OT shape mismatch
                          + lambda_mass * relative patch mass mismatch
                          + lambda_l1   * local mean absolute error

    Returns
        Mean weighted local discrepancy (lower is better).
        Mean local sqrt(OT) shape distance for reference.
    """
    x_true = np.asarray(x_true, dtype=np.float64)
    x_rec = np.asarray(x_rec, dtype=np.float64)

    if x_true.ndim != 2 or x_rec.ndim != 2:
        raise ValueError("patchwise_transport_discrepancy expects two 2D grayscale images")

    if x_true.shape != x_rec.shape:
        raise ValueError("x_true and x_rec must have the same shape")

    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be > 0")

    h, w = x_true.shape
    if h < patch_size or w < patch_size:
        raise ValueError("patch_size is larger than the image dimensions")

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
                        a=a,
                        b=b,
                        C=C,
                        reg=reg,
                        maxiter=maxiter,
                        tol=tol,
                        eps=eps,
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
