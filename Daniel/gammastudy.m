%% Gamma Study: Optimal gamma depends on noise level
clear; close all; clc;

imagePath = '/Users/danielbensimon/Downloads/testimages/manWithHat.tiff';

gamma_vals = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
noise_levels = [0.01, 0.05, 0.1, 0.2];

psnr_results = zeros(length(noise_levels), length(gamma_vals));

params.maxiter        = 500;
params.tprimaldr      = 0.5;
params.rhoprimaldr    = 1.0;
params.tol            = 1e-4;
params.record_metrics = true;

for ni = 1:length(noise_levels)
    [I_original, b, kernel] = buildCorruptedImage(imagePath, ...
        'BlurKind', 'gaussian', 'BlurSize', 9, 'BlurSigma', 2.0, ...
        'NoiseType', 'saltpepper', 'NoiseLevel', noise_levels(ni));

    params.I_reference = I_original;

    for gi = 1:length(gamma_vals)
        params.gammal1 = gamma_vals(gi);
        xinitial = zeros(size(b));
        [~, info] = primalDouglasRachford('l1', xinitial, kernel, b, params);

        psnr_results(ni, gi) = info.psnr(end);
        fprintf('noise=%.2f  gamma=%.3f  PSNR=%.2f dB\n', ...
            noise_levels(ni), gamma_vals(gi), psnr_results(ni, gi));
    end
end

figure;
for ni = 1:length(noise_levels)
    semilogx(gamma_vals, psnr_results(ni,:), 'o-', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('noise = %.2f', noise_levels(ni)));
    hold on;
end
xlabel('\gamma');
ylabel('PSNR (dB)');
legend('Location', 'best');
title('Effect of \gamma on Reconstruction Quality');
grid on;