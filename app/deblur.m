function x = deblur(problem, algorithm, xinitial, kernel, b, i)
% deblur  Entry-point function per MATH 563 project spec.
%
%   x = deblur(problem, algorithm, xinitial, kernel, b)
%   x = deblur(problem, algorithm, xinitial, kernel, b, i)
%
% INPUTS:
%   problem   - 'l1' or 'l2'  (data fidelity term)
%   algorithm - 'PDR', 'PDDR', 'ADMM', or 'CP'
%   xinitial  - (H x W) double, initial guess; pass [] to start from b
%   kernel    - (kH x kW) double, PSF (point-spread function)
%   b         - (H x W) double, blurred+noisy observation in [0,1]
%   i         - (optional) parameter struct with fields:
%                 gamma   - TV regularization weight
%                           default 0.012 for l2, 0.006 for l1
%                 t       - primal step size           (default 1.0; PDR: 0.25)
%                 rho     - over-relaxation / ADMM penalty (default 1.0; PDR: 1.25)
%                 s       - dual step size for CP      (default 0.25)
%                 maxiter - iteration cap              (default 500; ADMM: 200)
%                 tol     - convergence tolerance      (default 1e-4)
%
% OUTPUT:
%   x - (H x W) double, reconstructed image in [0,1]
%
% PRINTED SUMMARY (always):
%   Algorithm, problem type, iterations, final objective, CPU time, convergence.
%
% EXAMPLE:
%   psf = fspecial('gaussian', 9, 2);
%   b   = imfilter(im2double(imread('cameraman.tif')), psf, 'circular');
%   b   = b + 0.02 * randn(size(b));
%   i.gamma = 0.01; i.maxiter = 200;
%   x = deblur('l2', 'ADMM', [], psf, b, i);

% ---- Set up paths -----------------------------------------------------------
appDir = fileparts(mfilename('fullpath'));
if isempty(appDir), appDir = fileparts(which('deblur')); end
addpath(fullfile(appDir,'algorithms'));
addpath(fullfile(appDir,'prox_operators'));

% ---- Build config struct from parameter struct i ----------------------------
if nargin < 6 || isempty(i), i = struct(); end

config.problem = lower(problem);
if strcmp(config.problem, 'l1')
    config.gamma = fieldOrDefault(i, 'gamma', 0.006);
else
    config.gamma = fieldOrDefault(i, 'gamma', 0.012);
end
switch upper(algorithm)
    case 'PDR',  t_def = 0.25; rho_def = 1.25; maxiter_def = 500;
    case 'ADMM', t_def = 1.0;  rho_def = 1.0;  maxiter_def = 200;
    case 'CP',   t_def = 0.25; rho_def = 1.0;  maxiter_def = 500;
    otherwise,   t_def = 1.0;  rho_def = 1.0;  maxiter_def = 500;
end
config.t       = fieldOrDefault(i, 't',       t_def);
config.rho     = fieldOrDefault(i, 'rho',     rho_def);
config.s       = fieldOrDefault(i, 's',       0.25);
config.maxiter = fieldOrDefault(i, 'maxiter', maxiter_def);
config.tol     = fieldOrDefault(i, 'tol',     1e-4);
config.delta   = fieldOrDefault(i, 'delta',   0.1);
config.verbose = false;   % suppress per-iteration output; summary printed below

% ---- Pass caller's initial guess to the solver ------------------------------
if nargin >= 3 && ~isempty(xinitial)
    config.x_init = xinitial;
end

% ---- Dispatch to the appropriate solver -------------------------------------
switch upper(algorithm)
    case 'PDR',  [x, info] = PDR(b,  kernel, config);
    case 'PDDR', [x, info] = PDDR(b, kernel, config);
    case 'ADMM', [x, info] = ADMM(b, kernel, config);
    case 'CP',   [x, info] = CP(b,   kernel, config);
    otherwise
        error('deblur: unknown algorithm ''%s''. Choose: PDR, PDDR, ADMM, CP.', algorithm);
end

% ---- Required printed summary -----------------------------------------------
fprintf('\n=== Deblurring Results ===\n');
fprintf('  Algorithm : %s\n',    upper(algorithm));
fprintf('  Problem   : %s\n',    upper(problem));
fprintf('  Iterations: %d\n',    info.iterations);
fprintf('  Objective : %.6e\n',  info.objective(end));
fprintf('  CPU time  : %.3f s\n', info.time_sec);
fprintf('  Converged : %s\n',    mat2str(info.converged));
fprintf('==========================\n');

end


% ---- Local helpers ----------------------------------------------------------

function val = fieldOrDefault(s, field, default)
% Return s.field if it exists, else default.
if isfield(s, field)
    val = s.(field);
else
    val = default;
end
end
