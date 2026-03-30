import time
import numpy as np

# ---------------------------------------------------------------------------
# FFT-based operators (numpy.fft, periodic boundary conditions)
# ---------------------------------------------------------------------------

def _psf_to_otf(psf, shape):
    kH, kW = psf.shape
    H, W = shape
    padded = np.zeros((H, W), dtype=np.float64)
    padded[:kH, :kW] = psf
    padded = np.roll(padded, -(kH // 2), axis=0)
    padded = np.roll(padded, -(kW // 2), axis=1)
    return np.fft.fft2(padded)


def _blur(x, k_hat):
    return np.real(np.fft.ifft2(k_hat * np.fft.fft2(x)))


def _blur_adj(x, k_hat):
    return np.real(np.fft.ifft2(np.conj(k_hat) * np.fft.fft2(x)))


def _grad(x):
    gx = np.roll(x, -1, axis=1) - x
    gy = np.roll(x, -1, axis=0) - x
    return gx, gy


def _div(gx, gy):
    return (np.roll(gx, 1, axis=1) - gx) + (np.roll(gy, 1, axis=0) - gy)


def _A(x, k_hat):
    y1 = _blur(x, k_hat)
    y2, y3 = _grad(x)
    return y1, y2, y3


def _AT(y1, y2, y3, k_hat):
    return _blur_adj(y1, k_hat) + _div(y2, y3)


def _laplacian_symbol(shape):
    m, n = shape
    wx = 2 * np.pi * np.arange(n) / n
    wy = 2 * np.pi * np.arange(m) / m
    WX, WY = np.meshgrid(wx, wy)
    return (2 - 2 * np.cos(WX)) + (2 - 2 * np.cos(WY))


# ---------------------------------------------------------------------------
# Proximal operators
# ---------------------------------------------------------------------------

def _soft_thresh(x, lam):
    lam = max(lam, 0.0)
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _proj_box(x, lb=0.0, ub=1.0):
    return np.clip(x, lb, ub)


def _prox_l1_shifted(v, b, tau):
    return b + _soft_thresh(v - b, tau)


def _prox_l2_shifted(v, b, tau):
    return (v + 2.0 * tau * b) / (1.0 + 2.0 * tau)


def _prox_iso(vx, vy, lam, eps=1e-12):
    lam = max(lam, 0.0)
    r = np.sqrt(vx**2 + vy**2)
    scale = np.maximum(1.0 - lam / (r + eps), 0.0)
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

    Returns
    -------
    x_sol : float64 ndarray (H, W) -- reconstructed image
    info  : dict with keys
        'objective'   : list of objective values (one per iteration)
        'rel_change'  : list of relative iterate changes
        'iterations'  : number of iterations run
        'converged'   : bool
        'time_sec'    : wall-clock time
    """
    if problem not in ("l1", "l2"):
        raise ValueError("problem must be 'l1' or 'l2'.")
    if not (0 < rho < 2):
        raise ValueError("rho must satisfy 0 < rho < 2.")

    H, W = b.shape
    k_hat = _psf_to_otf(psf, b.shape)

    # Denominator for (I + t^2 * A^T A)^{-1} in Fourier domain
    # = 1 + t^2 * (|k_hat|^2 + laplacian)
    denom = 1.0 + t**2 * (np.abs(k_hat)**2 + _laplacian_symbol(b.shape))

    # --- Initialise primal-dual variables ---
    x0 = b.copy() if x0 is None else x0.copy()

    z = x0.copy()           # primal dual variable (same space as x)
    w1 = np.zeros((H, W))   # dual block 1  (Kx space)
    w2 = np.zeros((H, W))   # dual block 2  (D1 x space)
    w3 = np.zeros((H, W))   # dual block 3  (D2 x space)

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
        x_half = _proj_box(z)

        # Dual: prox_{t*g*}(w) = w - t * prox_{g/t}(w/t)  [Moreau identity]
        if problem == "l1":
            pg1 = _prox_l1_shifted(w1 / t, b, 1.0 / t)
        else:
            pg1 = _prox_l2_shifted(w1 / t, b, 1.0 / t)

        pg2, pg3 = _prox_iso(w2 / t, w3 / t, gamma / t)

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
        ATq = _AT(q1, q2, q3, k_hat)
        rhs = p - t * ATq
        x_new = np.real(np.fft.ifft2(np.fft.fft2(rhs) / denom))

        Ax1, Ax2, Ax3 = _A(x_new, k_hat)
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
        x_cur = _proj_box(z)
        rel_change = np.linalg.norm(x_cur - x_prev) / (np.linalg.norm(x_prev) + 1e-12)
        history_rel.append(rel_change)

        # Objective value
        Kx = _blur(x_cur, k_hat)
        gx, gy = _grad(x_cur)
        tv = float(np.sum(np.sqrt(gx**2 + gy**2)))
        if problem == "l1":
            fidelity = float(np.sum(np.abs(Kx - b)))
        else:
            fidelity = float(np.sum((Kx - b)**2))
        history_obj.append(fidelity + gamma * tv)

        if verbose and (k == 1 or k % print_every == 0):
            print(f"  iter {k:4d}  obj={history_obj[-1]:.6e}  rel_change={rel_change:.6e}")

        if rel_change < tol and k > 1:
            converged = True
            if verbose:
                print(f"  Converged at iteration {k}  (rel_change={rel_change:.2e})")
            break

        x_prev = x_cur

    x_sol = _proj_box(z)
    elapsed = time.time() - t_start

    if verbose:
        print(f"  Done — {k} iters, {elapsed:.2f}s, converged={converged}")

    info = {
        "objective":  history_obj,
        "rel_change": history_rel,
        "iterations": k,
        "converged":  converged,
        "time_sec":   elapsed,
    }
    return x_sol, info
