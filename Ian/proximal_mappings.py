import numpy as np

"""
This file contains the proximal operators

Courtney's objectivess are

    (l1 model)  min_x  ||Kx - b||_1   + delta_S(x) + gamma ||Dx||_iso
    (l2 model)  min_x  ||Kx - b||_2^2 + delta_S(x) + gamma ||Dx||_iso
under regularization constraint -
    S = { x : 0 <= x <= 1 }


In ADMM we use the same method as the PDRS, but now applied to the dual problem.
The splitting is done so that the resultant separation has functions whose proximal
mappings are computed easily
This file therefore implements proximal maps for
    1> delta_S(x)              -> projection onto [0,1] 
    2> ||y - b||_1             -> shifted soft-thresholding
    3> ||y - b||_2^2           -> quadratic prox (smooth)
    4> gamma * ||(p,q)||_iso   -> isotropic norm
"""
def soft_thresh(x, lam):
    """
    Elementwise

    For each entry x_i:
        soft(x_i; lam) = sign(x_i) * max(|x_i| - lam, 0)
        The explicit computation for this is in homework 3
    This is the proximal operator of
        lam * ||x||_1
    
    Where
    1> Inputs 
        i> x = ndarray
        ii> lambda = lam = float
    2> Output = Soft-Threshold ndarray
    """
    lam = max(lam, 0.0)
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


"""
Regularization Constraint - delta_S(x)"""
def proj_box(x, lb=0.0, ub=1.0):
    """
    The proximal operator of the indicator function delta_[lb,ub](x)
    is exactly Euclidean projection onto the box, which is simply clipping.
    lb, ub are the lower and upper bounds of the range(projection)
    """
    return np.clip(x, lb, ub)


def prox_indicator_box(v, tau=None, lb=0.0, ub=1.0):
    return proj_box(v, lb=lb, ub=ub)
