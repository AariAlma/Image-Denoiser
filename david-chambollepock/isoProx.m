function [p1, p2] = isoProx(x1, x2, lambda)
% isoProx: Proximal operator for Isotropic Total Variation (Block Soft-Thresholding).
% Solves argmin_u { lambda ||u||_2 + 0.5 ||u - x||_2^2 } for each pixel block.

    % Compute the L2 norm (magnitude) at each pixel
    magn = sqrt(x1.^2 + x2.^2);

    % Compute the scaling factor.
    % We add 'eps' to the denominator to prevent division-by-zero errors
    % at pixels where the gradient magnitude is exactly zero.
    scale = max(1 - lambda ./ (magn + eps), 0);

    % Apply the thresholding scale block-wise
    p1 = x1 .* scale;
    p2 = x2 .* scale;
end
