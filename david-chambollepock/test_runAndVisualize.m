% TEST_RUNANDVISUALIZE: Script to demonstrate how to use runAndVisualize.m
% Ensure this is saved in the same directory as runAndVisualize.m, optsolver.m,
% chambollepock.m, and your test images (cameraman.jpg, manWithHat.tiff).

clear; clc; close all;

fprintf('--- Setting up universal parameters ---\n');

% 1. Define Universal Parameters for the solver (struct 'i')
i.maxiter = 300;
i.tcp = 0.3;      % Primal step size (for Chambolle-Pock)
i.scp = 0.3;      % Dual step size (for Chambolle-Pock)
i.gammal1 = 0.05; % Regularization weight for L1 Problem
i.gammal2 = 0.02; % Regularization weight for L2 Problem

% 2. Define Blur Kernel
% 15x15 Gaussian blur with a standard deviation of 5
kernel = fspecial('gaussian', [15, 15], 5);

% -------------------------------------------------------------------------
% EXPERIMENT 1: Cameraman with Salt & Pepper Noise (L1 Penalty preferred)
% -------------------------------------------------------------------------
img_file1 = 'cameraman.jpg';
noise_type1 = 'salt & pepper';
noise_param1 = 0.05; % 5% pixel corruption

fprintf('\n--- Running Experiment 1 (L1 + Salt & Pepper) ---\n');
% Call the wrapper function from the Canvas
runAndVisualize(img_file1, 'l1', 'chambollepock', kernel, noise_type1, noise_param1, i);

% -------------------------------------------------------------------------
% EXPERIMENT 2: Man with Hat with Gaussian Noise (L2 Penalty preferred)
% -------------------------------------------------------------------------
img_file2 = 'manWithHat.tiff';
noise_type2 = 'gaussian';
noise_param2 = 0.005; % Gaussian variance

fprintf('\n--- Running Experiment 2 (L2 + Gaussian) ---\n');
% Call the wrapper function from the Canvas
runAndVisualize(img_file2, 'l2', 'chambollepock', kernel, noise_type2, noise_param2, i);

fprintf('\nAll experiments completed. Check the generated figures!\n');
