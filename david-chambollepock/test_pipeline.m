% TEST_PIPELINE: Script to test the Chambolle-Pock Algorithm
clear; clc; close all;

fprintf('Initializing Image Deblurring Pipeline...\n');

% 1. Load and prepare the true image
% Using MATLAB's built-in cameraman image
I_true = im2double(imread('manWithHat.tiff'));

% Normalize physical pixel constraints to [0, 1] as requested
mn = min(I_true(:));
mx = max(I_true(:));
I_true = (I_true - mn) / (mx - mn);

% 2. Generate the blurred and noisy observation (b)
% Create a Gaussian blur kernel
kernel = fspecial('gaussian', [15, 15], 5); % 15x15 kernel filter with std dev 5
% NOTE: The [15, 15] and 5 are just placeholder values that give a sensible
% blurred image for this prototype test. The project doc actually explicitly
% gives 15x15 as an example, but we're encouraged to try more.

% Blur the image using circular convolution to match our math models
b_blurred = imfilter(I_true, kernel, 'circular', 'conv');

% Add Salt & Pepper noise (particularly interesting for L1 data fidelity testing)
noiseDensity = 0.05;
b = imnoise(b_blurred, 'salt & pepper', noiseDensity);

% 3. Set Algorithm Parameters
% Note: ||A||_2^2 <= 9. For convergence we need s*t*9 < 1.
% s=0.3, t=0.3 gives 0.09 < 0.11, which guarantees mathematical stability.
% NOTE: We need to implement a check on the step size condition as a little
% bit of robustness to inappropriate inputs
i.maxiter = 300;
i.tcp = 0.3;     % Primal step size
i.scp = 0.3;     % Dual step size
i.gammal1 = 0.05; % Regularization weight for L1 problem
i.gammal2 = 0.02; % Regularization weight for L2 problem


% 4. Run the Solver
fprintf('Running Chambolle-Pock with L1 Data Fidelity (Salt & Pepper resistant)...\n');
% We initialize with the blurred image (b) to give the algorithm a head start
x_reconstructed = optsolver('l1', 'chambollepock', b, kernel, b, i);

% 5. Display the Results
figure('Name', 'Chambolle-Pock Deblurring Results', 'Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
imshow(I_true, []);
title('Original True Image');

subplot(1, 3, 2);
imshow(b, []);
title(sprintf('Blurred & Noisy\n(Gaussian + Salt&Pepper)'));

subplot(1, 3, 3);
imshow(x_reconstructed, []);
title(sprintf('Reconstructed Image\n(Chambolle-Pock L1)'));
