Configuration

"scale": 1.0,
        "problem": "l2",
        
        # Blurring
        "kernel": "gaussian",
        # Gaussian kernel parameters
        "ksize": 46,
        "sigma": 20.0,
        
        
        
        # Motion blur parameters
        "motion_length": 9,
        "motion_angle": 0.0,

        "noise": "gaussian",
        "noise_level": 0.01,

        #ADMM parameters
        "gamma": 0.0001,
        "t": 1.0,
        "rho": 1.0,
        "maxiter": 1000,
        "tol": 1e-10,

        # Randomness
        "seed": 0,

<img width="1200" height="400" alt="Figure_2" src="https://github.com/user-attachments/assets/828c9703-b4db-49fd-b2c1-f039dbd39ce7" />

