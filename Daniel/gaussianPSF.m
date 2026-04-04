function psf = gaussianPSF(sz, sigma)
%GAUSSIANPSF  Normalized 2-D Gaussian blur kernel.
%
%   psf = gaussianPSF(sz, sigma)
%
%   INPUTS
%     sz    - kernel side length (odd recommended)
%     sigma - standard deviation in pixels
%
%   OUTPUT
%     psf   - sz x sz double array, non-negative, sums to 1

half = floor(sz / 2);
[X, Y] = meshgrid(-half:half, -half:half);
psf = exp(-(X.^2 + Y.^2) / (2 * sigma^2));
psf = psf / sum(psf(:));
end