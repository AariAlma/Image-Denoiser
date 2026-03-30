import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from admm import admm_solve
from operators import blur
"""
1> Accept an input image path from the user
2> Convert the image to grayscale
3> Normalize and resize pixel values to [0, 1]
"""
"""image utilities"""
def load_grayscale_image(path):
    """
    Load an image from disk, convert it to grayscale, and normalize to [0,1].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = Image.open(path).convert("L")   # grayscale
    img = np.asarray(img, dtype=np.float64)

    # Normalize to [0,1]
    img -= img.min()
    max_val = img.max()
    if max_val > 0:
        img /= max_val

    return img

def resize_image(img, scale):
    """
    Resize image by a given scale factor.
    """
    if scale == 1.0:
        return img

    pil_img = Image.fromarray((255 * img).astype(np.uint8))
    new_size = (
        max(1, int(pil_img.size[0] * scale)),
        max(1, int(pil_img.size[1] * scale)),
    )
    pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    out = np.asarray(pil_img, dtype=np.float64) / 255.0
    return out
