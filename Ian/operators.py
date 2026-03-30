import numpy as np

from scipy.fft import ifft2, fft2
"""In this file I have defined all the operators to be used for this project"""

def psf_to_otf(psf, shape): 
    """
    This is for transforming the point-spread function into a pixel-cushioned version to deal with icruclar convolution arounf the edges
    """
    out = np.zeros(shape)
    kh, kw = psf.shape
    out[:kh, :kw] = psf
    
    out = np.roll(out, -(kh // 2), axis=0)
    out = np.roll(out, -(kw // 2), axis=1)
    return fft2(out)

""" 
Blurrring - The main idea here is to convolve the two matrices and then perform an inverse FFT to recover the
non-spectral signal """
#Kx = IFFT(K_hat, x_hat)
def blur(x, k_hat):
    return np.real(ifft2(k_hat * fft2(x)))
#KTx = IFFT(k_hat* . x_hat), k_hat* is the adjoint (Blur the adjoint too)
def blur_adj(x, k_hat): 
    return np.real(ifft2(np.conj(k_hat) * fft2(x)))

"""
Compute Gradients - both the length and breadth-wise comparison
This the discrete gradient operator D which captures local variation in pixel intensities
""" 
def grad(x):
    gx = np.roll(x, -1, axis=1) - x
    gy = np.roll(x, -1, axis=0) - x
    return gx, gy

"""
Divergence - This computes div = -nabla_transpose, and then
<grad(x), g> = -<x, div(g)>
"""
def div(gx, gy):
    return (np.roll(gx, 1, axis=1) - gx) + (np.roll(gy, 1, axis=0) - gy)

"""
Below are the matrices defined in the description as A,A_transpose -
In the description, the matrix A was defined as A = [K D]T
And on imposing the constraint Ax = (.), we get Ax = [Kx Dx]T = [Kx, grad(x)]
Thus, the elements within the matrix A are 

1> Kx = y1,
2> grad(x) = (y2,y3)

These have been linked to the placeholder function g(y1, (y2,y3)) to express the maximum monotone operator 
as a separable sum
"""
def A_apply(x, k_hat):
    y1 = blur(x, k_hat)
    y2, y3 = grad(x)
    return y1, y2, y3

"""
AT [y1, y2]T = KTy1 - div(y2)"""
def AT_apply(y1, y2, y3, k_hat):
    return blur_adj(y1, k_hat) + div(y2, y3)

"""
This computes the Fourier transform of the laplacian (nabla). 
Link - https://www.math.ucla.edu/~tao/preprints/fourier.pdf
"""
def laplacian_symbol(shape):
    m, n = shape
    wx = 2 * np.pi * np.arange(n) / n
    wy = 2 * np.pi * np.arange(m) / m
    WX, WY = np.meshgrid(wx, wy)
    return (2 - 2 * np.cos(WX)) + (2 - 2 * np.cos(WY))

"""
ATAx = Normal Equation 
"""
def normal_eq_symbol(k_hat, shape):
    return 1.0 + np.abs(k_hat)**2 + laplacian_symbol(shape)

def solve_normal(rhs, denom):
    return np.real(ifft2(fft2(rhs) / denom))
