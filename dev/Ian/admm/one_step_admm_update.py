"""One-step ADMM"""
def admm_step(state, b, k_hat, denom, gamma, t, rho, problem):
    """
        x^k = (I + A^T A)^(-1) [ u + A^T y - (1/t)( w + A^T z ) ]
        u^k = prox_{t^{-1} f}( rho x^k + (1-rho)u + w/t )
        y^k = prox_{t^{-1} g}( rho A x^k + (1-rho)y + z/t )
        w^k = w + t (x^k - u^k)
        z^k = z + t (A x^k - y^k)
    """
    x_old = state["x"]
    u_old = state["u"]
    w_old = state["w"]

    y1_old = state["y1"]
    y2_old = state["y2"]
    y3_old = state["y3"]

    z1_old = state["z1"]
    z2_old = state["z2"]
    z3_old = state["z3"]

    
    # x-update
    # Compute A^T y and A^T z once 
    ATy = AT_apply(y1_old, y2_old, y3_old, k_hat)
    ATz = AT_apply(z1_old, z2_old, z3_old, k_hat)

    rhs = u_old + ATy - (1.0 / t) * (w_old + ATz)
    x_new = solve_normal(rhs, denom)

    
    # u-update
    # Since f is the indicator of [0,1], its prox is just projection.
    u_arg = rho * x_new + (1.0 - rho) * u_old + w_old / t
    u_new = proj_box(u_arg)

    # y-update
    # First compute A x^k = (Kx, Dx).
    Ax1, Ax2, Ax3 = A_apply(x_new, k_hat)

    # Overrelaxed argument for the prox of g.
    y1_arg = rho * Ax1 + (1.0 - rho) * y1_old + z1_old / t
    y2_arg = rho * Ax2 + (1.0 - rho) * y2_old + z2_old / t
    y3_arg = rho * Ax3 + (1.0 - rho) * y3_old + z3_old / t

    # prox_{t^{-1} g}
    tau = 1.0 / t

    if problem == "l1":
        y1_new, y2_new, y3_new = prox_g_l1(
            y1_arg, y2_arg, y3_arg, b=b, gamma=gamma, tau=tau
        )
    elif problem == "l2":
        y1_new, y2_new, y3_new = prox_g_l2(
            y1_arg, y2_arg, y3_arg, b=b, gamma=gamma, tau=tau
        )
    else:
        raise ValueError("problem must be 'l1' or 'l2'.")

    
    
    
    # dual updates
    w_new = w_old + t * (x_new - u_new)
    z1_new = z1_old + t * (Ax1 - y1_new)
    z2_new = z2_old + t * (Ax2 - y2_new)
    z3_new = z3_old + t * (Ax3 - y3_new)

    #Verify
    primal_res_u = np.linalg.norm(x_new - u_new)
    primal_res_y = np.sqrt(
        np.linalg.norm(Ax1 - y1_new) ** 2
        + np.linalg.norm(Ax2 - y2_new) ** 2
        + np.linalg.norm(Ax3 - y3_new) ** 2
    )
    step_norm = np.linalg.norm(x_new - x_old)

    new_state = {
        "x": x_new,
        "u": u_new,
        "w": w_new,
        "y1": y1_new,
        "y2": y2_new,
        "y3": y3_new,
        "z1": z1_new,
        "z2": z2_new,
        "z3": z3_new,
    }

    info = {
        "primal_res_u": primal_res_u,
        "primal_res_y": primal_res_y,
        "step_norm": step_norm,
    
    }

    return new_state, info
