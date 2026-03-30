def show_results(original, blurred_noisy, reconstructed):
    """
    Show original, corrupted, and reconstructed images side-by-side.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(blurred_noisy, cmap="gray")
    plt.title("Blurred + Noisy")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("ADMM Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def show_rec(reconstructed):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("ADMM Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_image(img, path):
    """
    Save an image array in [0,1] to disk.
    """
    img_uint8 = np.clip(255 * img, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)
