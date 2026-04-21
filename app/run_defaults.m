% run_defaults.m
% Spec-mandated "run every algorithm with default parameters" driver
% (see project_description_2026.pdf §Project grading).
%
% Running this script:
%   1. builds a blurred + noisy test image from input_images/cameraman.jpg,
%   2. calls deblur() once per algorithm on the L1 problem (1),
%   3. calls deblur() once per algorithm on the L2 problem (2),
% using the spec-compliant long algorithm names and qualified parameter
% field names. Per-iteration output and an end-of-run summary are printed
% for each algorithm.
%
% The grader may swap in any starting point by editing the single
% assignment `x = b;` below. To use a different test image, edit the
% `imagePath` variable.

% ---- Paths ------------------------------------------------------------------
appDir = fileparts(mfilename('fullpath'));
addpath(fullfile(appDir, 'algorithms'));
addpath(fullfile(appDir, 'prox_operators'));
addpath(fullfile(appDir, 'corrupting'));
addpath(fullfile(appDir, 'metrics'));

% ---- Build a corrupted test image ------------------------------------------
imagePath = fullfile(appDir, 'input_images', 'cameraman.jpg');
[xTrue, b, kernel] = buildCorruptedImage(imagePath, ...
    'BlurKind',   'gaussian', ...
    'BlurSize',   9, ...
    'BlurSigma',  2.0, ...
    'NoiseType',  'gaussian', ...
    'NoiseLevel', 0.02, ...
    'Seed',       0);  %#ok<ASGLU>

% ---- Starting point (grader may replace this line) --------------------------
x = b;

% ---- Common parameters ------------------------------------------------------
i = struct();
i.maxiter = 500;

% ---- L1 problem: run every algorithm ---------------------------------------
fprintf('\n################ L1 problem (||Kx - b||_1 + delta_S + gamma ||Dx||_iso) ################\n');

i.gammal1 = 0.006;

% Primal Douglas-Rachford
i.tprimaldr    = 0.25;
i.rhoprimaldr  = 1.25;
x_l1_pdr       = deblur('l1', 'douglasrachfordprimal',     x, kernel, b, i);

% Primal-Dual Douglas-Rachford
i.tprimaldualdr    = 1.0;
i.rhoprimaldualdr  = 1.0;
x_l1_pddr          = deblur('l1', 'douglasrachfordprimaldual', x, kernel, b, i);

% ADMM
i.tadmm   = 1.0;
i.rhoadmm = 1.0;
x_l1_admm = deblur('l1', 'admm',                           x, kernel, b, i);

% Chambolle-Pock
i.tcp    = 0.25;
i.scp    = 0.25;
x_l1_cp  = deblur('l1', 'chambollepock',                   x, kernel, b, i);

% ---- L2 problem: run every algorithm ---------------------------------------
fprintf('\n################ L2 problem (||Kx - b||_2^2 + delta_S + gamma ||Dx||_iso) ################\n');

i.gammal2 = 0.012;

% Primal Douglas-Rachford
i.tprimaldr    = 0.25;
i.rhoprimaldr  = 1.25;
x_l2_pdr       = deblur('l2', 'douglasrachfordprimal',     x, kernel, b, i);

% Primal-Dual Douglas-Rachford
i.tprimaldualdr    = 1.0;
i.rhoprimaldualdr  = 1.0;
x_l2_pddr          = deblur('l2', 'douglasrachfordprimaldual', x, kernel, b, i);

% ADMM
i.tadmm   = 1.0;
i.rhoadmm = 1.0;
x_l2_admm = deblur('l2', 'admm',                           x, kernel, b, i);

% Chambolle-Pock
i.tcp    = 0.25;
i.scp    = 0.25;
x_l2_cp  = deblur('l2', 'chambollepock',                   x, kernel, b, i);

fprintf('\nAll 8 algorithm/problem combinations completed.\n');
