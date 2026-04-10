import time
import numpy as np

# Optional GPU backend (CuPy). Falls back to NumPy silently if unavailable.
try:
    import cupy as _cupy
    _HAS_CUPY = True
except ImportError:
    _cupy = None
    _HAS_CUPY = False


def _get_xp(use_gpu):
    """Return cupy if GPU is requested and available, else numpy."""
    if use_gpu and _HAS_CUPY:
        return _cupy
    return np


# ---------------------------------------------------------------------------
# FFT-based operators (periodic boundary conditions)
# ---------------------------------------------------------------------------

def _psf_to_otf(psf, shape, xp):
    kH, kW = psf.shape
    H, W = shape
    padded = xp.zeros((H, W), dtype=xp.float64)
    padded[:kH, :kW] = psf
    padded = xp.roll(padded, -(kH // 2), axis=0)
    padded = xp.roll(padded, -(kW // 2), axis=1)
    return xp.fft.fft2(padded)


def _blur(x, k_hat, xp):
    return xp.real(xp.fft.ifft2(k_hat * xp.fft.fft2(x)))


def _blur_adj(x, k_hat, xp):
    return xp.real(xp.fft.ifft2(xp.conj(k_hat) * xp.fft.fft2(x)))


def _grad(x, xp):
    gx = xp.roll(x, -1, axis=1) - x
    gy = xp.roll(x, -1, axis=0) - x
    return gx, gy


def _div(gx, gy, xp):
    return (xp.roll(gx, 1, axis=1) - gx) + (xp.roll(gy, 1, axis=0) - gy)


def _A(x, k_hat, xp):
    y1 = _blur(x, k_hat, xp)
    y2, y3 = _grad(x, xp)
    return y1, y2, y3


def _AT(y1, y2, y3, k_hat, xp):
    return _blur_adj(y1, k_hat, xp) + _div(y2, y3, xp)


def _laplacian_symbol(shape, xp):
    m, n = shape
    wx = 2 * xp.pi * xp.arange(n) / n
    wy = 2 * xp.pi * xp.arange(m) / m
    WX, WY = xp.meshgrid(wx, wy)
    return (2 - 2 * xp.cos(WX)) + (2 - 2 * xp.cos(WY))


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------

def _soft_thresh(x, lam, xp):
    lam = max(lam, 0.0)
    return xp.sign(x) * xp.maximum(xp.abs(x) - lam, 0.0)


def _proj_box(x, xp, lb=0.0, ub=1.0):
    return xp.clip(x, lb, ub)


def _prox_l1_shifted(v, b, tau, xp):
    return b + _soft_thresh(v - b, tau, xp)


def _prox_l2_shifted(v, b, tau, xp):
    return (v + 2.0 * tau * b) / (1.0 + 2.0 * tau)


def _prox_huber_shifted(v, b, tau, delta, xp):
    """prox_{tau * phi_delta(. - b)}(v)  —  Huber data fidelity.

    phi_delta(r) = r²/(2·delta)  if |r| ≤ delta  (L2)
                 = |r| - delta/2  if |r| > delta  (L1)

    Closed-form prox (elementwise):
        |c| ≤ delta + tau  →  c · delta / (delta + tau)   (L2 shrinkage)
        |c| > delta + tau  →  sign(c) · (|c| - tau)       (soft threshold)
    """
    c = v - b
    abs_c = xp.abs(c)
    transition = delta + tau
    r = xp.where(
        abs_c <= transition,
        c * delta / transition,
        xp.sign(c) * xp.maximum(abs_c - tau, 0.0),
    )
    return b + r


def _prox_iso(vx, vy, lam, xp, eps=1e-12):
    lam = max(lam, 0.0)
    r = xp.sqrt(vx**2 + vy**2)
    scale = xp.maximum(1.0 - lam / (r + eps), 0.0)
    return scale * vx, scale * vy


# ---------------------------------------------------------------------------
# Primal-Dual Douglas-Rachford solver
# ---------------------------------------------------------------------------

def primal_dual_dr_solve(
    b, psf,
    x0=None,
    problem="l2",
    gamma=0.05,
    t=1.0,
    rho=1.0,
    maxiter=500,
    tol=1e-4,
    verbose=True,
    print_every=50,
    use_gpu=False,
    huber_delta=0.1,
):
    """
    Primal-Dual Douglas-Rachford splitting for image deblurring / denoising.

    Solves:
        (L1)  min_x  ||Kx - b||_1   + delta_S(x) + gamma ||Dx||_iso
        (L2)  min_x  ||Kx - b||_2^2 + delta_S(x) + gamma ||Dx||_iso
    where S = {x : 0 <= x <= 1}.

    The problem is cast as  min_x f(x) + g(Ax)  with:
        f  = delta_S        (box indicator, prox = proj_box)
        g  = data_fidelity + gamma * ||.||_iso   (separable over A's output blocks)
        A  = [K; D1; D2]   (blur stacked with discrete gradient)

    Douglas-Rachford is applied in the product (primal x dual) space to:
        0 in M(x,y) + N(x,y)
        M(x,y) = (df(x), dg*(y))   -- separable, resolved via Moreau
        N(x,y) = (A^T y, -Ax)      -- skew-symmetric, resolved via FFT linear solve

    Parameters
    ----------
    b        : float64 ndarray (H, W)  -- blurred + noisy observation, values in [0,1]
    psf      : float64 ndarray (kH,kW) -- spatial-domain blur kernel
    x0       : float64 ndarray (H, W) or None  -- initial guess (default: b)
    problem  : str  -- 'l1' or 'l2'
    gamma    : float -- TV regularization weight
    t        : float -- step size  (t > 0)
    rho      : float -- DR relaxation parameter  (0 < rho < 2)
    maxiter  : int   -- maximum number of iterations
    tol      : float -- relative-change convergence tolerance
    verbose  : bool  -- print iteration log
    print_every : int -- log frequency
    use_gpu  : bool  -- run FFT and array ops on GPU via CuPy (ignored if CuPy
                        is not installed)

    Returns
    -------
    x_sol : float64 ndarray (H, W) -- reconstructed image (always on CPU)
    info  : dict with keys
        'objective'   : list of objective values (one per iteration)
        'rel_change'  : list of relative iterate changes
        'iterations'  : number of iterations run
        'converged'   : bool
        'time_sec'    : wall-clock time
        'used_gpu'    : bool
    """
    if problem not in ("l1", "l2", "huber"):
        raise ValueError("problem must be 'l1', 'l2', or 'huber'.")
    if not (0 < rho < 2):
        raise ValueError("rho must satisfy 0 < rho < 2.")

    xp = _get_xp(use_gpu)
    on_gpu = (xp is not np)

    if verbose and on_gpu:
        print("  [GPU] Running on CUDA via CuPy.")
    elif verbose and use_gpu and not _HAS_CUPY:
        print("  [GPU] CuPy not available — falling back to CPU.")

    # Move inputs to the chosen device
    b   = xp.asarray(b,   dtype=xp.float64)
    psf = xp.asarray(psf, dtype=xp.float64)

    H, W = b.shape
    k_hat = _psf_to_otf(psf, b.shape, xp)

    # Denominator for (I + t^2 * A^T A)^{-1} in Fourier domain
    denom = 1.0 + t**2 * (xp.abs(k_hat)**2 + _laplacian_symbol(b.shape, xp))

    # --- Initialise primal-dual variables ---
    if x0 is None:
        x0 = b.copy()
    else:
        x0 = xp.asarray(x0, dtype=xp.float64)

    z  = x0.copy()
    w1 = xp.zeros((H, W), dtype=xp.float64)
    w2 = xp.zeros((H, W), dtype=xp.float64)
    w3 = xp.zeros((H, W), dtype=xp.float64)

    history_obj = []
    history_rel = []
    converged = False
    x_prev = x0.copy()

    t_start = time.time()

    for k in range(1, maxiter + 1):

        # ------------------------------------------------------------------
        # Step 1: resolvent of t*M  (separate primal and dual blocks)
        # ------------------------------------------------------------------

        # Primal: prox_{t*f}(z) = proj_box(z)
        x_half = _proj_box(z, xp)

        # Dual: prox_{t*g*}(w) = w - t * prox_{g/t}(w/t)  [Moreau identity]
        if problem == "l1":
            pg1 = _prox_l1_shifted(w1 / t, b, 1.0 / t, xp)
        elif problem == "l2":
            pg1 = _prox_l2_shifted(w1 / t, b, 1.0 / t, xp)
        else:  # huber
            pg1 = _prox_huber_shifted(w1 / t, b, 1.0 / t, huber_delta, xp)

        pg2, pg3 = _prox_iso(w2 / t, w3 / t, gamma / t, xp)

        v1_half = w1 - t * pg1
        v2_half = w2 - t * pg2
        v3_half = w3 - t * pg3

        # ------------------------------------------------------------------
        # Step 2: DR reflection of M
        # ------------------------------------------------------------------
        p  = 2.0 * x_half - z
        q1 = 2.0 * v1_half - w1
        q2 = 2.0 * v2_half - w2
        q3 = 2.0 * v3_half - w3

        # ------------------------------------------------------------------
        # Step 3: resolvent of t*N
        #   Solve (I + t^2 A^T A) x_new = p - t * A^T q
        # ------------------------------------------------------------------
        ATq = _AT(q1, q2, q3, k_hat, xp)
        rhs = p - t * ATq
        x_new = xp.real(xp.fft.ifft2(xp.fft.fft2(rhs) / denom))

        Ax1, Ax2, Ax3 = _A(x_new, k_hat, xp)
        y1_new = q1 + t * Ax1
        y2_new = q2 + t * Ax2
        y3_new = q3 + t * Ax3

        # ------------------------------------------------------------------
        # Step 4: relaxed dual update
        # ------------------------------------------------------------------
        z  += rho * (x_new  - x_half)
        w1 += rho * (y1_new - v1_half)
        w2 += rho * (y2_new - v2_half)
        w3 += rho * (y3_new - v3_half)

        # ------------------------------------------------------------------
        # Convergence check
        # ------------------------------------------------------------------
        x_cur = _proj_box(z, xp)
        rel_change = float(xp.linalg.norm(x_cur - x_prev) / (xp.linalg.norm(x_prev) + 1e-12))
        history_rel.append(rel_change)

        # Objective value
        Kx = _blur(x_cur, k_hat, xp)
        gx, gy = _grad(x_cur, xp)
        tv = float(xp.sum(xp.sqrt(gx**2 + gy**2)))
        if problem == "l1":
            fidelity = float(xp.sum(xp.abs(Kx - b)))
        elif problem == "l2":
            fidelity = float(xp.sum((Kx - b)**2))
        else:  # huber
            r = xp.abs(Kx - b)
            fidelity = float(xp.sum(xp.where(
                r <= huber_delta,
                r**2 / (2.0 * huber_delta),
                r - huber_delta / 2.0,
            )))
        history_obj.append(fidelity + gamma * tv)

        if verbose and (k == 1 or k % print_every == 0):
            print(f"  iter {k:4d}  obj={history_obj[-1]:.6e}  rel_change={rel_change:.6e}")

        # Iterate-change criterion (works well for L2)
        iterate_converged = rel_change < tol and k > 1

        # Objective-plateau criterion (robust for L1 non-smooth oscillations):
        # DR iterates can persistently oscillate above tol for non-smooth objectives
        # even when near the solution; check if the objective has stopped improving.
        _OBJ_WINDOW = 20
        obj_plateau = False
        if len(history_obj) >= _OBJ_WINDOW:
            obj_win = history_obj[-_OBJ_WINDOW:]
            obj_range = max(obj_win) - min(obj_win)
            obj_plateau = obj_range / (abs(history_obj[-1]) + 1e-12) < tol

        if iterate_converged or obj_plateau:
            converged = True
            if verbose:
                print(f"  Converged at iteration {k}  (rel_change={rel_change:.2e})")
            break

        x_prev = x_cur

    x_sol = _proj_box(z, xp)

    # Always return a CPU numpy array
    if on_gpu:
        x_sol = xp.asnumpy(x_sol)

    elapsed = time.time() - t_start

    if verbose:
        print(f"  Done — {k} iters, {elapsed:.2f}s, converged={converged}")

    info = {
        "objective":  history_obj,
        "rel_change": history_rel,
        "iterations": k,
        "converged":  converged,
        "time_sec":   elapsed,
        "used_gpu":   on_gpu,
    }
    return x_sol, info
