function x_prox = boxProx(x)
% boxProx - Projection onto the interval [0,1]
%
% DESCRIPTION:
%   Projects each element of x onto [0,1]:
%   - If x_i is already in [0,1], keep it unchanged
%   - If x_i < 0, set x_i = 0
%   - If x_i > 1, set x_i = 1
%
% INPUT:
%   x - Input array (any size)
%
% OUTPUT:
%   x_prox - Projected array (same size as x)
%
% USAGE:
%   x_prox = boxProx(x)

x_prox = max(0, min(1, x));

end

% tests
boxProx([-0.5, 0.3, 0.7, 1.2])

boxProx([0, 0.5, 1])  % Boundary values

