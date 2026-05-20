function p = l1Prox(x, lambda)
% l1Prox: Proximal operator for the L1 norm (1D Soft-Thresholding).
% Solves argmin_u { lambda ||u||_1 + 0.5 ||u - x||_2^2 }

    p = sign(x) .* max(abs(x) - lambda, 0);
end
