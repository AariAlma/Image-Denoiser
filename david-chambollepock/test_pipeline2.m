% TEST_PIPELINE: Script to test the Chambolle-Pock Algorithm directly
clear; clc; close all;

fprintf('Initializing Image Deblurring Pipeline...\n');

% 1. Load and prepare the true image
I_true = imread('cameraman.jpg');
if size(I_true, 3) == 3
    I_true = rgb2gray(I_true);
end
I_true = double(I_true);

% Normalize physical pixel constraints to [0, 1]
mn = min(I_true(:));
mx = max(I_true(:));
I_true = (I_true - mn) / (mx - mn);

% 2. Generate the blurred and noisy observation
kernel = fspecial('gaussian', [15, 15], 5);
b_blurred = imfilter(I_true, kernel, 'circular', 'conv');

% Add Salt & Pepper noise
noiseDensity = 0.05;
b = imnoise(b_blurred, 'salt & pepper', noiseDensity);

% 3. Set Algorithm Parameters
i.maxiter = 300;
i.tcp = 0.3;
i.scp = 0.3;
i.gammal1 = 0.05;
i.gammal2 = 0.02;

% Specify the filename where the plot should be saved
convergence_plot_name = 'CP_Convergence_L1_Cameraman.png';

% 4. Run the Solver (Passing the filename as the 7th argument)
fprintf('Running Chambolle-Pock...\n');
x_reconstructed = optsolver('l1', 'chambollepock', b, kernel, b, i, convergence_plot_name);

% 5. Display the Visual Results
figure('Name', 'Chambolle-Pock Deblurring Results', 'Position', [100, 100, 1200, 400]);
subplot(1, 3, 1); imshow(I_true, []); title('Original True Image');
subplot(1, 3, 2); imshow(b, []); title('Blurred & Noisy');
subplot(1, 3, 3); imshow(x_reconstructed, []); title('Reconstructed (L1)');
