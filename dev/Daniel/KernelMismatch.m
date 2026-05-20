%% Kernel Mismatch Experiment
clear; close all; clc;

imagePath = '/Users/danielbensimon/Downloads/testimages/manWithHat.tiff';

% True blur: Gaussian with sigma = 5
[I_original, b, ~] = buildCorruptedImage(imagePath, ...
    'BlurKind', 'gaussian', 'BlurSize', 15, 'BlurSigma', 5.0, ...
    'NoiseType', 'gaussian', 'NoiseLevel', 0.01);

% Sweep over assumed sigma values (some match, most don't)
sigma_assumed = 2:0.5:8;
psnr_results  = zeros(size(sigma_assumed));
ssim_results  = zeros(size(sigma_assumed));

params.maxiter        = 500;
params.gammal1        = 0.01;
params.tprimaldr      = 0.5;
params.rhoprimaldr    = 1.0;
params.tol            = 1e-4;
params.record_metrics = true;
params.I_reference    = I_original;

for si = 1:length(sigma_assumed)
    wrong_kernel = gaussianPSF(15, sigma_assumed(si));
    xinitial = zeros(size(b));
    [xSol, info] = primalDouglasRachford('l1', xinitial, wrong_kernel, b, params);

    psnr_results(si) = info.psnr(end);
    ssim_results(si) = info.ssim(end);
    fprintf('sigma_assumed=%.1f  PSNR=%.2f  SSIM=%.4f\n', ...
        sigma_assumed(si), psnr_results(si), ssim_results(si));
end

figure;
yyaxis left;  plot(sigma_assumed, psnr_results, 'o-', 'LineWidth', 1.5);
ylabel('PSNR (dB)');
yyaxis right; plot(sigma_assumed, ssim_results, 's-', 'LineWidth', 1.5);
ylabel('SSIM');
xlabel('Assumed \sigma');
xline(5, '--k', 'LineWidth', 1.5);
title('Kernel Mismatch: Gaussian \sigma (True \sigma = 5)');
grid on;

%% Motion Kernel Mismatch: Angle Sweep
clear; close all; clc;

imagePath = '/Users/danielbensimon/Downloads/testimages/manWithHat.tiff';

true_angle = 30;
true_len   = 20;

[I_original, b, ~] = buildCorruptedImage(imagePath, ...
    'BlurKind', 'motion', 'MotionLen', true_len, 'MotionAngle', true_angle, ...
    'NoiseType', 'gaussian', 'NoiseLevel', 0.01);

angles_assumed = 0:5:60;
psnr_angle = zeros(size(angles_assumed));
ssim_angle = zeros(size(angles_assumed));

params.maxiter        = 500;
params.gammal1        = 0.01;
params.tprimaldr      = 0.5;
params.rhoprimaldr    = 1.0;
params.tol            = 1e-4;
params.record_metrics = true;
params.I_reference    = I_original;

for ai = 1:length(angles_assumed)
    wrong_kernel = motionPSF(true_len, angles_assumed(ai));
    xinitial = zeros(size(b));
    [~, info] = primalDouglasRachford('l1', xinitial, wrong_kernel, b, params);

    psnr_angle(ai) = info.psnr(end);
    ssim_angle(ai) = info.ssim(end);
    fprintf('angle_assumed=%5.1f  PSNR=%.2f  SSIM=%.4f\n', ...
        angles_assumed(ai), psnr_angle(ai), ssim_angle(ai));
end

figure;
yyaxis left;  plot(angles_assumed, psnr_angle, 'o-', 'LineWidth', 1.5);
ylabel('PSNR (dB)');
yyaxis right; plot(angles_assumed, ssim_angle, 's-', 'LineWidth', 1.5);
ylabel('SSIM');
xlabel('Assumed Angle (degrees)');
xline(true_angle, '--k', 'True angle', 'LineWidth', 1.5);
title('Motion Kernel Mismatch: Angle');
grid on;