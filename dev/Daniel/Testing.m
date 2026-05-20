%% Main Script
clear; close all; clc;

imagePath = ['/Users/danielbensimon/Downloads/testimages/manwithhat.tiff'];

% Step 1: corrupt the image
[I_original, b, kernel] = buildCorruptedImage(imagePath, ...
    'BlurKind',    'motion',       ...
    'MotionLen',   40,             ...
    'MotionAngle', 40,             ...
    'NoiseType',   'saltpepper',   ...
    'NoiseLevel',  0.1);
disp(size(I_original))
disp(size(b))
% Step 2: set solver parameters
params.maxiter        = 200;
params.tprimaldr      = 0.25;
params.rhoprimaldr    = 1.25;
params.gammal1        = 0.001;
params.tol            = 1e-3;
params.record_metrics = true;
params.I_reference    = I_original;

% Step 3: run solver
xinitial = zeros(size(b));
[xSol, info] = primalDouglasRachford('l1', xinitial, kernel, b, params);

% Step 4: display images
figure('Name', 'Restoration Result');
subplot(1, 3, 1); imshow(I_original, []); title('Original');
subplot(1, 3, 2); imshow(b, []);          title('Blurred & Noisy');
subplot(1, 3, 3); imshow(xSol, []);       title('Restored');

function plotConvergence(info)

if isempty(info.objective)
    warning('No metrics recorded. Set params.record_metrics = true.');
    return;
end

iters = 1:numel(info.objective);

figure('Name', 'Convergence Plots', 'NumberTitle', 'off');

%--- Subplot 1: Objective function ---
subplot(1, 3, 1);
loglog(iters, info.objective, 'b-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Objective value');
title('Objective Function Value');
grid on;

%--- Subplot 2: PSNR ---
subplot(1, 3, 2);
if ~isempty(info.psnr)
    plot(iters, info.psnr, 'g-', 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('PSNR (dB)');
    title('PSNR vs. Ground Truth');
    grid on;
else
    text(0.5, 0.5, 'No PSNR data (I\_reference not provided)', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    title('PSNR vs. Ground Truth');
    axis off;
end

%--- Subplot 3: SSIM ---
subplot(1, 3, 3);
if ~isempty(info.ssim)
    plot(iters, info.ssim, 'r-', 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('SSIM');
    title('SSIM vs. Ground Truth');
    ylim([0 1]);
    grid on;
else
    text(0.5, 0.5, 'No SSIM data (I\_reference not provided)', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    title('SSIM vs. Ground Truth');
    axis off;
end

sgtitle(sprintf('Primal Douglas-Rachford Convergence (%d iterations)', ...
    info.iterations), 'FontSize', 13, 'FontWeight', 'bold');
end


% Step 5: plot convergence
plotConvergence(info);