def main():
    config = {
        #"image": "manWithHat.tiff",
        "image": "cameraman.jpg",
        #"image": "mcgill.jpg",

        "scale": 1.0,

        #   "l1" -> ||Kx-b||_1 + gamma TV(x)
        #   "l2" -> ||Kx-b||_2^2 + gamma TV(x)
        "problem": "l1",


        # "kernel" = 
        #   1> "gaussian"
        #   2> "motion"
        "kernel": "gaussian",

        # Gaussian kernel parameters
        "ksize": 9,
        "sigma": 2.0,

        # Motion blur parameters
        "motion_length": 9,
        "motion_angle": 0.0,

        # Choice of Noise
        #   "gaussian"
        #   "saltpepper"
        "noise": "gaussian",
        "noise_level": 0.01,

        #ADMM parameters
        "gamma": 0.006,
        "t": 1.0,
        "rho": 1.0,
        "maxiter": 1000,
        "tol": 1e-10,

        # Randomness
        "seed": 0,

        "save": None,

        # Print solver progress
        "verbose": True,
        #Print status interval
        "print_every": 10,
    }

# Preprocessing
    x_true = load_grayscale_image(config["image"])
    x_true = resize_image(x_true, config["scale"])

    
    #Blurring
    psf = build_kernel(
        kind=config["kernel"],
        size=config["ksize"],
        sigma=config["sigma"],
        motion_length=config["motion_length"],
        motion_angle=config["motion_angle"],
    )

    # Corrupted Image
    from operators import psf_to_otf

    k_hat = psf_to_otf(psf, x_true.shape)
    b_blur = blur(x_true, k_hat)
    b = add_noise(
        b_blur,
        noise_type=config["noise"],
        noise_level=config["noise_level"],
        seed=config["seed"],
    )

    
    # RUN ADMM
    x_rec, history = admm_solve(
        b=b,
        psf=psf,
        x0=b,
        problem=config["problem"],
        gamma=config["gamma"],
        t=config["t"],
        rho=config["rho"],
        maxiter=config["maxiter"],
        tol=config["tol"],
        verbose=config["verbose"],
        print_every=config["print_every"],
    )


    show_results(x_true, b, x_rec)
    show_rec(x_rec)

#Logged Outputs
    print("\nRun complete.")
    print(f"Input image         : {config['image']}")
    print(f"Problem             : {config['problem']}")
    print(f"Kernel              : {config['kernel']}")
    print(f"Noise               : {config['noise']}")
    print(f"Gamma               : {config['gamma']}")
    print(f"t                   : {config['t']}")
    print(f"rho                 : {config['rho']}")
    print(f"Final objective     : {history['objective'][-1]:.6e}")
    print(f"Iterations recorded : {len(history['objective'])}")


if __name__ == "__main__":
    main()
