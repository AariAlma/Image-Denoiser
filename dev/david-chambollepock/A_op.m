function [Kx, D1x, D2x] = A_op(x, kernel)
% A_op: Computes the forward linear operator A = [K; D]
%
% INPUTS:
%   x      : Current image iterate
%   kernel : Blur kernel
%
% OUTPUTS:
%   Kx     : Convolved image
%   D1x    : Horizontal forward finite differences
%   D2x    : Vertical forward finite differences

    % K: Convolution with circular boundary conditions.
    % We specify 'conv' to ensure it uses strict convolution,
    % making its adjoint the cross-correlation.
    Kx = imfilter(x, kernel, 'circular', 'conv');

    % D: Forward finite differences using circular shifts
    D1x = circshift(x, [0, -1]) - x; % Horizontal derivative
    D2x = circshift(x, [-1, 0]) - x; % Vertical derivative
end
