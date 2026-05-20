function val = computeSSIM(x, xhat)
% computeSSIM  Global (non-windowed) Structural Similarity Index.
%
% Uses the standard SSIM formula (Wang et al. 2004) computed over the
% entire image rather than a sliding window.  This is O(N^2) and fast
% enough to call every solver iteration.
%
% For color (H×W×3) inputs, returns the mean SSIM across the three channels.
% Constants C1, C2 follow the standard L=1 convention (pixels in [0,1]).
%
% INPUTS:
%   x    - (H x W) or (H x W x 3) double, ground truth image in [0,1]
%   xhat - same size as x, reconstructed image
%
% OUTPUT:
%   val  - scalar in (-1, 1]; higher is better; 1 = identical images

% Color branch: average SSIM over R, G, B channels independently
if size(x, 3) == 3
    val = mean([computeSSIM(x(:,:,1), xhat(:,:,1)), ...
                computeSSIM(x(:,:,2), xhat(:,:,2)), ...
                computeSSIM(x(:,:,3), xhat(:,:,3))]);
    return
end

C1 = (0.01)^2;   % stabilises luminance term
C2 = (0.03)^2;   % stabilises contrast/structure terms

x    = x(:);
xhat = xhat(:);

mu_x    = mean(x);
mu_xhat = mean(xhat);

sig_x    = std(x);
sig_xhat = std(xhat);
sig_cross = mean((x - mu_x) .* (xhat - mu_xhat));

val = ((2*mu_x*mu_xhat + C1) * (2*sig_cross + C2)) / ...
      ((mu_x^2 + mu_xhat^2 + C1) * (sig_x^2 + sig_xhat^2 + C2));
end
