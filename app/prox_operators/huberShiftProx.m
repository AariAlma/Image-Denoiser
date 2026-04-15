function y_prox = huberShiftProx(y, b, t, delta)
% huberShiftProx  Proximal operator of  t * phi_delta(. - b)
%
%   phi_delta(r) = r^2 / 2                      if |r| <= delta  (quadratic)
%                = delta * (|r| - delta/2)       if |r| >  delta  (linear)
%
% The proximal operator prox_{t * phi_delta(. - b)}(y):
%
%   Let u = y - b.  Then:
%     if |u| <= (1+t)*delta  :  x = b + u / (1+t)         [L2 shrink toward b]
%     if |u| >  (1+t)*delta  :  x = y - t*delta*sign(u)   [L1 soft-shift]
%
% This smoothly interpolates between an L2-like shrink (for small residuals)
% and an L1-like soft-threshold (for large residuals), matching the Huber loss.
%
% INPUTS
%   y      - array of any size, the point to evaluate at
%   b      - same size as y (or scalar), the observation shift
%   t      - positive scalar, the proximal step size
%   delta  - positive scalar, the Huber transition threshold
%
% OUTPUT
%   y_prox - same size as y

u      = y - b;
thresh = (1 + t) * delta;
in_l2  = abs(u) <= thresh;
y_prox = b + in_l2 .* (u / (1 + t)) + (~in_l2) .* (u - t * delta * sign(u));
end
