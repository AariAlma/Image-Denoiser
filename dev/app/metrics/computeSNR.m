function val = computeSNR(x, xhat)
% computeSNR  Signal-to-Noise Ratio in dB.
%
% SNR = 10 * log10( E[x^2] / E[(x - xhat)^2] )
%
% Measures the ratio of signal power (ground truth) to error power.
% Unlike PSNR, the numerator uses the actual signal energy rather than
% the peak value, so SNR is sensitive to image brightness.
%
% INPUTS:
%   x    - (H x W) double, ground truth image in [0,1]
%   xhat - (H x W) double, reconstructed image
%
% OUTPUT:
%   val  - scalar in dB; higher is better

signal_power = mean(x(:).^2);
noise_power  = mean((x(:) - xhat(:)).^2);

if noise_power < eps
    val = Inf;
else
    val = 10 * log10(signal_power / noise_power);
end
end
