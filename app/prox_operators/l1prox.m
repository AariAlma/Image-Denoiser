function x_prox = l1Prox(x, t)
% Soft thresholding
% (x_prox)_i = x_i - t if x_i > t
% (x_prox)_i = x_i + t if x_i < -t
% (x_prox)_i = 0 if abs(x_i) <= t

x_prox = sign(x) .* max(abs(x) - t, 0);

end
