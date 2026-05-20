if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 4))
    b = rng.standard_normal((4, 4))
    vx = rng.standard_normal((4, 4))
    vy = rng.standard_normal((4, 4))
    print("proj_box shape", proj_box(x).shape)
    print("soft_thresh shape", soft_thresh(x, 0.1).shape)
    print("prox_l1_shifted shape", prox_l1_shifted(x, b, 0.2).shape)
    print("prox_l2_shifted shape", prox_l2_shifted(x, b, 0.2).shape)
    px, py = prox_iso(vx, vy, 0.3)
    print("prox_iso shapes", px.shape, py.shape)

"""
This is a small check of dimensionalities
"""
