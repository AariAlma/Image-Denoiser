function Aty = At_op(y1, y2, y3, kernel)
% At_op: Computes the adjoint linear operator A^T * y
% Evaluates K^T*y1 + D1^T*y2 + D2^T*y3
%
% INPUTS:
%   y1, y2, y3 : Dual variables
%   kernel     : Blur kernel
%
% OUTPUT:
%   Aty        : Result of the adjoint operator mapped back to image space

    % K^T: The adjoint of circular convolution is circular cross-correlation.
    % We achieve this by rotating the kernel 180 degrees.
    kernel_adj = rot90(kernel, 2);
    Kty1 = imfilter(y1, kernel_adj, 'circular', 'conv');

    % D^T: The adjoint of forward finite differences is backward differences.
    % We achieve this by shifting in the opposite (positive) direction.
    D1ty2 = circshift(y2, [0, 1]) - y2;
    D2ty3 = circshift(y3, [1, 0]) - y3;

    % Sum all adjoint components to form A^T y
    Aty = Kty1 + D1ty2 + D2ty3;
end
