def objective_value(x, b, k_hat, gamma, problem):
    r = blur(x, k_hat) - b
    # If L1
    if problem == "l1":
        fidelity = np.sum(np.abs(r))
    # If Quadtratic
    elif problem == "l2":
        fidelity = np.sum(r**2)
    else:
        raise ValueError("problem must be 'l1' or 'l2'.")
    return fidelity + gamma * tv_isotropic(x)
