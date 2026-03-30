function p = boxProx(x)
% boxProx: Computes the Euclidean projection onto the unit box [0, 1].
% This serves as the proximal operator for the indicator function \delta_C(x).

    p = max(0, min(1, x));
end
