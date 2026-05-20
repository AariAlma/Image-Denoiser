% TEST_PIPELINE: Script to test the Chambolle-Pock Algorithm directly
clear; clc; close all;

rng(2026);

fprintf('Initializing Image Deblurring Pipeline...\n');

% ---------------------------------------------------------
% 1. Load and prepare the true image
% ---------------------------------------------------------
img_file = 'cameraman.jpg';
[~, img_name, ~] = fileparts(img_file);

I_true = imread(img_file);
if size(I_true, 3) == 3
    I_true = rgb2gray(I_true);
end
I_true = double(I_true);

% Normalize physical pixel constraints to [0, 1]
mn = min(I_true(:));
mx = max(I_true(:));
I_true = (I_true - mn) / (mx - mn);

% ---------------------------------------------------------
% 2. Define Degradation Parameters (Blur & Noise)
% ---------------------------------------------------------
kernel_type = 'Gaussian';
k_size = 15;
k_sigma = 5;
kernel = fspecial(lower(kernel_type), [k_size, k_size], k_sigma);

noise_type = 'salt & pepper';
noise_param = 0.05;

% Apply Degradation
b_blurred = imfilter(I_true, kernel, 'circular', 'conv');
b = imnoise(b_blurred, noise_type, noise_param);

% ---------------------------------------------------------
% 3. Set Algorithm Parameters
% ---------------------------------------------------------
problem_type = 'l1'; % 'l1' or 'l2'

i.maxiter = 300;
i.tcp = 0.3;
i.scp = 0.3;
i.gammal1 = 0.05;
i.gammal2 = 0.03;

% Determine active gamma for the title
if strcmp(problem_type, 'l1')
    gamma_val = i.gammal1;
else
    gamma_val = i.gammal2;
end

% Specify the filename where the convergence plot should be saved
convergence_plot_name = sprintf('cp_convergence_%s_%s.png', img_name, problem_type);

% ---------------------------------------------------------
% 4. Run the Solver
% ---------------------------------------------------------
fprintf('Running Chambolle-Pock...\n');
x_reconstructed = optsolver(problem_type, 'chambollepock', b, kernel, b, i, convergence_plot_name);

% ---------------------------------------------------------
% 5. Display the Visual Results with Parameter Annotations
% ---------------------------------------------------------
% Made the figure slightly taller (450) to make room for the super title
figure('Name', 'Chambolle-Pock Deblurring Results', 'Position', [100, 100, 1200, 600]);

subplot(1, 3, 1);
imshow(I_true, []);
title('Original True Image');

subplot(1, 3, 2);
imshow(b, []);
title(sprintf('Blurred & Noisy\n(%s + %s)', kernel_type, noise_type));

subplot(1, 3, 3);
imshow(x_reconstructed, []);
title(sprintf('Reconstructed\n(Chambolle-Pock %s)', upper(problem_type)));

% Add a Super Title containing all pipeline parameters
param_str = sprintf(['Pipeline Parameters:\n' ...
    'Degradation: %s Kernel [%dx%d], \\sigma=%.1f  |  Noise: %s (param=%.3f)\n' ...
    'Solver: Chambolle-Pock (%s)  |  Iters: %d, t=%.2f, s=%.2f, \\gamma=%.3f'], ...
    kernel_type, k_size, k_size, k_sigma, noise_type, noise_param, ...
    upper(problem_type), i.maxiter, i.tcp, i.scp, gamma_val);

% Cross-compatibility check for MATLAB vs GNU Octave
if exist('sgtitle', 'file') || exist('sgtitle', 'builtin')
    sgtitle(param_str, 'FontSize', 12, 'FontWeight', 'bold');
else
    % Fallback for GNU Octave / older MATLAB versions
    % Create an invisible axes over the entire figure and place text at the top
    axes('Position', [0 0 1 1], 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1]);
    text(0.5, 0.96, param_str, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'top', 'FontSize', 12, 'FontWeight', 'bold');
end
