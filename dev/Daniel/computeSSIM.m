function ssimVal = computeSSIM(x, xHat)
% COMPUTESSIM - Computes SSIM between two images using equation (35)
%
% INPUTS
%   x    - original clean image (normalized to [0,1])
%   xHat - reconstructed image  (normalized to [0,1])
%
% OUTPUT
%   ssimVal - scalar in [0,1], higher is better

% Flatten to vectors
x    = x(:);
xHat = xHat(:);
n    = length(x);

% Constants (standard choice C3 = C2/2 is absorbed into simplified form)
C1 = 0.01^2;   % stabilizes luminance
C2 = 0.03^2;   % stabilizes contrast

% --- Statistical quantities (equations 28, 29, 30) ---
mu_x    = mean(x);
mu_xHat = mean(xHat);

sigma_x    = sum((x    - mu_x).^2)    / (n - 1);
sigma_xHat = sum((xHat - mu_xHat).^2) / (n - 1);
sigma_xxHat = sum((x - mu_x) .* (xHat - mu_xHat)) / (n - 1);

% --- Simplified SSIM formula (equation 35) ---
ssimVal = (2*mu_x*mu_xHat + C1) * (2*sigma_xxHat + C2) / ...
          ((mu_x^2 + mu_xHat^2 + C1) * (sigma_x + sigma_xHat + C2));
end