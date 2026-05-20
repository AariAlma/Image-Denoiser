function x = deblur(problem, algorithm, xinitial, kernel, b, i)
% deblur  Entry-point function per MATH 563 project spec.
%
%   x = deblur(problem, algorithm, xinitial, kernel, b)
%   x = deblur(problem, algorithm, xinitial, kernel, b, i)
%
% INPUTS:
%   problem   - 'l1' or 'l2' (data fidelity term); 'huber' also supported
%   algorithm - spec-compliant long names or short aliases (case-insensitive):
%                 'douglasrachfordprimal'     | 'PDR'
%                 'douglasrachfordprimaldual' | 'PDDR'
%                 'admm'                      | 'ADMM'
%                 'chambollepock'             | 'CP'
%   xinitial  - (H x W) double, initial guess; pass [] to start from b
%   kernel    - (kH x kW) double, PSF (point-spread function)
%   b         - (H x W) double, blurred+noisy observation in [0,1]
%   i         - (optional) parameter struct. The spec-mandated fields are
%               read first, with a generic fallback for backwards
%               compatibility:
%                 maxiter           iteration limit                 (default 500)
%                 gammal1 / gammal2 TV weight for L1 / L2 problem   (0.006 / 0.012)
%                 tprimaldr    / rhoprimaldr     PDR  step / relaxation (0.25 / 1.25)
%                 tprimaldualdr/ rhoprimaldualdr PDDR step / relaxation (1.0  / 1.0 )
%                 tadmm        / rhoadmm         ADMM step / relaxation (1.0  / 1.0 )
%                 tcp          / scp             CP primal / dual step  (computed from ||A||)
%                 tol               convergence tolerance           (default 1e-4)
%                 delta             Huber transition threshold      (default 0.1)
%                 verbose           per-iteration printing          (default true)
%               Generic aliases (gamma, t, rho, s) are accepted when the
%               qualified field for the selected algorithm is absent.
%
% OUTPUT:
%   x - (H x W) double, reconstructed image in [0,1]
%
% PRINTED OUTPUT:
%   Per-iteration progress (iteration number and objective) plus an
%   end-of-run summary with final objective, iterations used, CPU time, and
%   convergence status.
%
% EXAMPLE (spec-compliant call):
%   psf = gaussianPSF(9, 2.0);
%   b   = imfilter(im2double(imread('input_images/cameraman.jpg')), psf, 'circular');
%   b   = b + 0.02 * randn(size(b));
%   i.maxiter = 500; i.gammal1 = 0.049;
%   i.tprimaldr = 2.0; i.rhoprimaldr = 0.1;
%   x = deblur('l1', 'douglasrachfordprimal', [], psf, b, i);

% ---- Set up paths -----------------------------------------------------------
appDir = fileparts(mfilename('fullpath'));
if isempty(appDir), appDir = fileparts(which('deblur')); end
addpath(fullfile(appDir,'algorithms'));
addpath(fullfile(appDir,'prox_operators'));

% ---- Normalize algorithm name to a canonical short form ---------------------
% Accept both spec-mandated long names and short aliases, case-insensitively.
canonAlgo = normalizeAlgorithm(algorithm);

% ---- Build config struct from parameter struct i ----------------------------
if nargin < 6 || isempty(i), i = struct(); end

config.problem = lower(problem);

% Per-algorithm defaults — keyed by the canonical short name.
switch canonAlgo
    case 'PDR',  t_def = 0.25; rho_def = 1.25; maxiter_def = 500;
        tField   = 'tprimaldr';      rhoField = 'rhoprimaldr';
    case 'PDDR', t_def = 1.0;  rho_def = 1.0;  maxiter_def = 500;
        tField   = 'tprimaldualdr';  rhoField = 'rhoprimaldualdr';
    case 'ADMM', t_def = 1.0;  rho_def = 1.0;  maxiter_def = 200;
        tField   = 'tadmm';          rhoField = 'rhoadmm';
    case 'CP',   t_def = [];   rho_def = 1.0;  maxiter_def = 500;  % CP.m computes 1/||A||_2
        tField   = 'tcp';            rhoField = '';                 % rho unused by CP
end

% gamma: prefer problem-qualified spec field (gammal1 / gammal2), then the
% generic 'gamma' fallback, then a problem-aware default.
if strcmp(config.problem, 'l1')
    gamma_def       = 0.006;
    gammaFieldSpec  = 'gammal1';
elseif strcmp(config.problem, 'l2')
    gamma_def       = 0.012;
    gammaFieldSpec  = 'gammal2';
else  % huber or any other fidelity: spec doesn't mandate a qualified name
    gamma_def       = 0.006;
    gammaFieldSpec  = '';
end
config.gamma = firstDefined(i, {gammaFieldSpec, 'gamma'}, gamma_def);

% t: algorithm-qualified spec field, then generic 't', then default.
if ~isempty(t_def)
    config.t = firstDefined(i, {tField, 't'}, t_def);
elseif isfield(i, tField)
    config.t = i.(tField);
elseif isfield(i, 't')
    config.t = i.t;   % CP: only forward if user explicitly specified
end

% rho: algorithm-qualified spec field, then generic 'rho', then default.
% (rhoField is empty for CP since the spec does not define one.)
if ~isempty(rhoField)
    config.rho = firstDefined(i, {rhoField, 'rho'}, rho_def);
else
    config.rho = firstDefined(i, {'rho'}, rho_def);
end

% s (Chambolle-Pock dual step): qualified scp, then generic s. If absent,
% CP.m computes a safe default from ||A||_2 — preserve that behavior by
% only forwarding the field when the user has actually set one.
if isfield(i, 'scp')
    config.s = i.scp;
elseif isfield(i, 's')
    config.s = i.s;
end

config.maxiter = firstDefined(i, {'maxiter'}, maxiter_def);
config.tol     = firstDefined(i, {'tol'},     1e-4);
config.delta   = firstDefined(i, {'delta'},   0.1);
config.verbose = firstDefined(i, {'verbose'}, true);   % per-iteration printing on by default (spec requirement)

% ---- Pass caller's initial guess to the solver ------------------------------
if nargin >= 3 && ~isempty(xinitial)
    config.x_init = xinitial;
end

% ---- Dispatch to the appropriate solver -------------------------------------
switch canonAlgo
    case 'PDR',  [x, info] = PDR(b,  kernel, config);
    case 'PDDR', [x, info] = PDDR(b, kernel, config);
    case 'ADMM', [x, info] = ADMM(b, kernel, config);
    case 'CP',   [x, info] = CP(b,   kernel, config);
end

% ---- Required printed summary -----------------------------------------------
fprintf('\n=== Deblurring Results ===\n');
fprintf('  Algorithm : %s\n',    canonAlgo);
fprintf('  Problem   : %s\n',    upper(problem));
fprintf('  Iterations: %d\n',    info.iterations);
fprintf('  Objective : %.6e\n',  info.objective(end));
fprintf('  CPU time  : %.3f s\n', info.time_sec);
fprintf('  Converged : %s\n',    mat2str(info.converged));
fprintf('==========================\n');

end


% ---- Local helpers ----------------------------------------------------------

function canon = normalizeAlgorithm(algorithm)
% Map any accepted alias for an algorithm (spec-compliant long name or short
% form, any casing) to the canonical internal short name used for dispatch.
% Unknown strings throw an error whose message contains the substrings
% 'PDR', 'ADMM', and 'CP' (required by test_deblur.m:S4).
if ~(ischar(algorithm) || (isstring(algorithm) && isscalar(algorithm)))
    error('deblur: ''algorithm'' must be a string. Choose: PDR/douglasrachfordprimal, PDDR/douglasrachfordprimaldual, ADMM/admm, CP/chambollepock.');
end
key = lower(strtrim(char(algorithm)));
switch key
    case {'pdr',  'douglasrachfordprimal'},     canon = 'PDR';
    case {'pddr', 'douglasrachfordprimaldual'}, canon = 'PDDR';
    case {'admm'},                              canon = 'ADMM';
    case {'cp',   'chambollepock'},             canon = 'CP';
    otherwise
        error('deblur: unknown algorithm ''%s''. Choose: PDR/douglasrachfordprimal, PDDR/douglasrachfordprimaldual, ADMM/admm, CP/chambollepock.', algorithm);
end
end

function val = firstDefined(s, fields, default)
% Return s.(fields{k}) for the first field name present on struct s; if
% none are set, return default. Empty entries in `fields` are skipped so
% callers can disable a slot by passing ''.
for k = 1:numel(fields)
    f = fields{k};
    if ~isempty(f) && isfield(s, f)
        val = s.(f);
        return;
    end
end
val = default;
end
