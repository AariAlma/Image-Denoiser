function val = computeMSE(x, xhat)
% computeMSE  Mean Squared Error between two images.
%
% MSE(x, xhat) = (1/N^2) * sum_{i,j} (x_{ij} - xhat_{ij})^2
%
% INPUTS:
%   x    - (H x W) double, ground truth image
%   xhat - (H x W) double, reconstructed image
%
% OUTPUT:
%   val  - scalar >= 0; lower is better

val = mean((x(:) - xhat(:)).^2);
end
