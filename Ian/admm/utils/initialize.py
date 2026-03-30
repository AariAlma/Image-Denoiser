def initialize_admm(x0, k_hat):
    """
    Initialize all ADMM variables.

    Variables:
        x, u, w        : image-shaped arrays
        y1, y2, y3     : blocks for Ax = (Kx, Dx)
        z1, z2, z3     : dual variables for Ax = y
    Store all the iterations of ADMM in a dictionary
    """
    x = x0.copy()
    u = x0.copy()
    w = np.zeros_like(x0)

    y1, y2, y3 = A_apply(x0, k_hat)
    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y2)
    z3 = np.zeros_like(y3)

    return {
        "x": x,
        "u": u,
        "w": w,
        "y1": y1,
        "y2": y2,
        "y3": y3,
        "z1": z1,
        "z2": z2,
        "z3": z3,
    }
