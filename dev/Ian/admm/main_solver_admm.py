def admm_solve(
    b,
    psf,
    x0=None,
    problem="l2",
    gamma=0.05,
    t=1.0,
    rho=1.0,
    maxiter=200,
    tol=1e-4,
    verbose=True,
    print_every=10,
):
    """
    Solve the deblurring / denoising problem using ADMM.

    Parameters
    ----------
    b is Observed blurred/noisy image.
    psf is Blur kernel in spatial domain.
    x0 is Initial guess. If None, use b.
    problem is Either 'l1' or 'l2'.
    gamma : is TV regularization parameter.
    t is ADMM step size.
    rho is Relaxation parameter.  0 < rho < 2
    """
    if x0 is None:
        x0 = b.copy()

    #Blur FFT
    k_hat = psf_to_otf(psf, b.shape)
    denom = normal_eq_symbol(k_hat, b.shape)

    # Initialize state
    state = initialize_admm(x0, k_hat)

    history = {
        "objective": [],
        "primal_res_u": [],
        "primal_res_y": [],
        "step_norm": [],
        "time_sec": [],
    }

    start = time.time()

    if verbose:
        print("Starting ADMM")
        print(f"problem = {problem}, gamma = {gamma}, t = {t}, rho = {rho}, maxiter = {maxiter}")

    
    # Main loop
    for k in range(1, maxiter + 1):
        state, info = admm_step(
            state=state,
            b=b,
            k_hat=k_hat,
            denom=denom,
            gamma=gamma,
            t=t,
            rho=rho,
            problem=problem,
        )

        x = state["x"]

        obj = objective_value(x, b, k_hat, gamma, problem)
        elapsed = time.time() - start

        history["objective"].append(obj)
        history["primal_res_u"].append(info["primal_res_u"])
        history["primal_res_y"].append(info["primal_res_y"])
        history["step_norm"].append(info["step_norm"])
        history["time_sec"].append(elapsed)

        if verbose and (k == 1 or k % print_every == 0 or k == maxiter):
            print(
                f"iter {k:4d} | "
                f"obj = {obj:.6e} | "
                f"||x-u|| = {info['primal_res_u']:.3e} | "
                f"||Ax-y|| = {info['primal_res_y']:.3e} | "
                f"step = {info['step_norm']:.3e}"
            )

        # stopping rule = primal consistency and x-step size are small (<= epsilon=tol)
        if max(info["primal_res_u"], info["primal_res_y"], info["step_norm"]) < tol:
            if verbose:
                print(f"Converged at iteration {k}.")
            break

    total_time = time.time() - start

    if verbose:
        print("\nADMM Reconstruction Outputs")
        print(f"status          : {'converged' if k < maxiter else 'maxiter reached'}")
        print(f"final iteration : {k}")
        print(f"final objective : {history['objective'][-1]:.6e}")
        print(f"cpu time (sec)  : {total_time:.3f}")

    return state["x"], history
