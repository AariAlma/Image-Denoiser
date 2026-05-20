function psnr_val = computePSNR(xTrue, xRecon)
% computePSNR  Peak Signal-to-Noise Ratio between two images in [0,1].
%
%   psnr_val = computePSNR(xTrue, xRecon)
%
%   INPUTS
%     xTrue  - double H×W, ground truth image, values in [0,1]
%     xRecon - double H×W, reconstructed image, values in [0,1]
%
%   OUTPUT
%     psnr_val - scalar PSNR in dB (Inf if images are identical)
%
%   Formula:  PSNR = 10 * log10( peak^2 / MSE )
%   With peak = 1 (images normalized to [0,1]), this simplifies to:
%             PSNR = 10 * log10( 1 / MSE )

mse = mean((xTrue(:) - xRecon(:)).^2);   % mean squared error, scalar

if mse == 0
    psnr_val = Inf;   % perfect reconstruction
else
    psnr_val = 10 * log10(1 / mse);   % peak is 1 for [0,1] images
end

end
