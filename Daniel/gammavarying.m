%% Gamma vs Noise Level: Image Grid
clear; close all; clc;

imagePath = '/Users/danielbensimon/Downloads/testimages/manwithhat.tiff';

gamma_vals   = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0];
noise_levels = [0.01, 0.05, 0.15, 0.3];

nG = length(gamma_vals);
nN = length(noise_levels);

params.maxiter        = 500;
params.tprimaldr      = 0.25;
params.rhoprimaldr    = 1.25;
params.tol            = 1e-4;
params.record_metrics = true;

restored  = cell(nN, nG);
blurred   = cell(nN, 1);
psnr_grid = zeros(nN, nG);

for ni = 1:nN
    [I_original, b, kernel] = buildCorruptedImage(imagePath, ...
        'BlurKind',    'motion',       ...
        'MotionLen',   50,             ...
        'MotionAngle', 50,             ...
        'NoiseType',   'saltpepper',   ...
        'NoiseLevel',  noise_levels(ni));

    blurred{ni} = b;
    params.I_reference = I_original;

    for gi = 1:nG
        params.gammal1 = gamma_vals(gi);
        xinitial = zeros(size(b));
        [xSol, info] = primalDouglasRachford('l1', xinitial, kernel, b, params);

        restored{ni, gi} = xSol;
        psnr_grid(ni, gi) = info.psnr(end);
        fprintf('noise=%.2f  gamma=%.3f  PSNR=%.2f dB\n', ...
            noise_levels(ni), gamma_vals(gi), psnr_grid(ni, gi));
    end
end

%% Plot
figure('Name', 'Gamma vs Noise Level', 'Position', [50 50 1400 800]);

for ni = 1:nN
    subplot(nN, nG + 1, (ni-1)*(nG+1) + 1);
    imshow(blurred{ni}, []);
    if ni == 1, title('Corrupted', 'FontSize', 9); end
    ylabel(sprintf('noise = %.2f', noise_levels(ni)), ...
        'FontSize', 9, 'FontWeight', 'bold');

    for gi = 1:nG
        subplot(nN, nG + 1, (ni-1)*(nG+1) + 1 + gi);
        imshow(restored{ni, gi}, []);
        if ni == 1
            title(sprintf('\\gamma = %.3f', gamma_vals(gi)), 'FontSize', 9);
        end
        xlabel(sprintf('%.1f dB', psnr_grid(ni, gi)), 'FontSize', 8);
    end
end

sgtitle('Effect of \gamma across Noise Levels (Salt & Pepper, \ell_1)', ...
    'FontSize', 13, 'FontWeight', 'bold');