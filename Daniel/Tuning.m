%% Parameter Tuning: sweep over rho and t
clear; close all; clc;

imagePath = '/Users/danielbensimon/Downloads/testimages/manWithHat.tiff';

[I_original, b, kernel] = buildCorruptedImage(imagePath, ...
    'BlurKind',   'gaussian',  ...
    'BlurSize',   10,           ...
    'BlurSigma',  2.0,         ...
    'NoiseType',  'saltpepper',...
    'NoiseLevel', 0.02);

% --- Define parameter grids ---
rho_vals = [0.25, 0.5, 0.75, 1.0, 1.5];
t_vals   = [0.25, 0.5, 0.75, 1.0, 1.5];

% --- Preallocate results ---
finalPSNR = zeros(length(rho_vals), length(t_vals));
numIters  = zeros(length(rho_vals), length(t_vals));

% --- Base params ---
params.maxiter        = 500;
params.gammal1        = 0.01;
params.tol            = 1e-3;
params.record_metrics = true;
params.I_reference    = I_original;

% --- Sweep ---
for ri = 1:length(rho_vals)
    for ti = 1:length(t_vals)
        params.rhoprimaldr = rho_vals(ri);
        params.tprimaldr   = t_vals(ti);

        xinitial = zeros(size(b));
        [~, info] = primalDouglasRachford('l1', xinitial, kernel, b, params);

        finalPSNR(ri, ti) = info.psnr(end);
        numIters(ri, ti)  = info.iterations;

        fprintf('rho=%.2f  t=%.2f  ->  PSNR=%.2f dB  iters=%d\n', ...
            rho_vals(ri), t_vals(ti), finalPSNR(ri, ti), numIters(ri, ti));
    end
end

% --- Plot PSNR heatmap ---
figure('Name', 'Parameter Tuning');

subplot(1, 2, 1);
imagesc(t_vals, rho_vals, finalPSNR);
colorbar;
xlabel('t'); ylabel('rho');
title('Final PSNR (dB)');
xticks(t_vals); yticks(rho_vals);
axis xy;

subplot(1, 2, 2);
imagesc(t_vals, rho_vals, numIters);
colorbar;
xlabel('t'); ylabel('rho');
title('Iterations to Convergence');
xticks(t_vals); yticks(rho_vals);
axis xy;

sgtitle('Parameter Tuning: \rho and t', 'FontSize', 13, 'FontWeight', 'bold');

% --- Print best combination ---
[bestPSNR, idx] = max(finalPSNR(:));
[bestRi, bestTi] = ind2sub(size(finalPSNR), idx);
fprintf('\nBest combination:\n');
fprintf('  rho = %.2f\n', rho_vals(bestRi));
fprintf('  t   = %.2f\n', t_vals(bestTi));
fprintf('  PSNR = %.2f dB\n', bestPSNR);