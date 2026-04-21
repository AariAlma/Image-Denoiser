% test_deblur.m
%
% Comprehensive test suite for deblur() — the teacher-spec entry point.
% Tests all algorithm/problem combos, output validity, parameter handling,
% edge-case inputs/PSFs, error behaviour, and reconstruction quality.
%
% Run from project/app/:
%   cd D:/school/math563/project/app
%   test_deblur
%
% Deblur's own printed summaries are suppressed via evalc so only
% [PASS]/[FAIL] lines appear.  A final table summarises results.

clear; clc;

% ── Paths ─────────────────────────────────────────────────────────────────────
thisDir = fileparts(which('test_deblur.m'));
if isempty(thisDir), thisDir = pwd; end
addpath(thisDir);
addpath(fullfile(thisDir, 'algorithms'));
addpath(fullfile(thisDir, 'prox_operators'));
addpath(fullfile(thisDir, 'corrupting'));
addpath(fullfile(thisDir, 'metrics'));

fprintf('\ntest_deblur.m  —  %s\n', char(datetime('now')));
fprintf('%s\n\n', repmat('─', 1, 72));

% ── Tracking ──────────────────────────────────────────────────────────────────
nPass    = 0;
nFail    = 0;
failList = {};

% ── Synthetic fixtures ────────────────────────────────────────────────────────
rng(0);
H = 32; W = 32;

% Piecewise-smooth ground-truth (gradient + step edge)
[Xg, Yg] = meshgrid(linspace(0,1,W), linspace(0,1,H));
x_true = max(0, min(1,  0.2 + 0.5*Xg + 0.2*(Yg > 0.5)));

% Standard 9x9 Gaussian PSF
psf_std = gaussianPSF(9, 2.0);

% Blurred + Gaussian noisy observation
b_std = imfilter(x_true, psf_std, 'circular') + 0.03*randn(H, W);
b_std = max(0, min(1, b_std));

% Salt-and-pepper noisy observation — used for L1 quality tests in S14
b_sp_base = imfilter(x_true, psf_std, 'circular');
sp_mask   = rand(H, W);
b_sp      = b_sp_base;
b_sp(sp_mask < 0.025) = 0;    % pepper (2.5%)
b_sp(sp_mask > 0.975) = 1;    % salt   (2.5%)

% Fast config: 5 iterations, no early stop — for speed in non-quality tests
FAST = struct('maxiter', 5, 'tol', 0);

% ── Inline test macro ─────────────────────────────────────────────────────────
% Pattern used throughout:
%   try
%       evalc("...deblur call, assigns result to workspace variable...");
%       assert(condition, 'message');
%       nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
%   catch e
%       nFail = nFail+1; failList{end+1} = tname;
%       fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
%   end
%
% evalc suppresses deblur's printed summary while still executing in the
% caller's workspace, so variables assigned inside are visible afterwards.

% =============================================================================
fprintf('── S1: All algorithm × problem combinations ─────────────────────────────\n');
% =============================================================================
% Every valid (problem, algorithm) pair must return an image with the right
% size, values in [0,1], all finite, and real-valued.

ALGS     = {'PDR', 'PDDR', 'ADMM', 'CP'};
PROBLEMS = {'l2',  'l1'};

for pi = 1:numel(PROBLEMS)
    for ai = 1:numel(ALGS)
        prob = PROBLEMS{pi};
        alg  = ALGS{ai};
        tname = sprintf('S1: %s + %s', prob, alg);
        try
            evalc("x = deblur(prob, alg, [], psf_std, b_std, FAST)");
            assert(isequal(size(x), [H, W]),        'wrong output size');
            assert(all(x(:) >= 0 & x(:) <= 1),     'output outside [0,1]');
            assert(all(isfinite(x(:))),             'non-finite values in output');
            assert(isreal(x),                       'complex values in output');
            nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
        catch e
            nFail = nFail+1; failList{end+1} = tname;
            fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
        end
    end
end

% =============================================================================
fprintf('\n── S2: Algorithm string case insensitivity ──────────────────────────────\n');
% =============================================================================
% deblur() calls upper(algorithm), so any casing must be accepted.

alg_variants = {'pdr', 'Pddr', 'admm', 'Cp', 'pDR', 'aDmM'};
for k = 1:numel(alg_variants)
    av    = alg_variants{k};
    tname = sprintf('S2: algorithm ''%s'' accepted', av);
    try
        evalc("x = deblur('l2', av, [], psf_std, b_std, FAST)");
        assert(all(isfinite(x(:))) && all(x(:) >= 0 & x(:) <= 1));
        nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
    catch e
        nFail = nFail+1; failList{end+1} = tname;
        fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
    end
end

% =============================================================================
fprintf('\n── S3: Problem string case insensitivity ────────────────────────────────\n');
% =============================================================================
% deblur() calls lower(problem) before storing in config, so 'L1'/'L2' must work.

prob_variants = {'L2', 'L1', 'l2', 'l1'};
for k = 1:numel(prob_variants)
    pv    = prob_variants{k};
    tname = sprintf('S3: problem ''%s'' accepted', pv);
    try
        evalc("x = deblur(pv, 'PDDR', [], psf_std, b_std, FAST)");
        assert(all(isfinite(x(:))) && all(x(:) >= 0 & x(:) <= 1));
        nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
    catch e
        nFail = nFail+1; failList{end+1} = tname;
        fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
    end
end

% =============================================================================
fprintf('\n── S4: Error handling ───────────────────────────────────────────────────\n');
% =============================================================================

% Unknown algorithm must throw
tname = 'S4: unknown algorithm throws error';
try
    evalc("deblur('l2', 'BOGUS', [], psf_std, b_std, FAST)");
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s  (no error was thrown)\n', tname);
catch
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
end

% Error message must name the valid choices
tname = 'S4: error message names PDR/PDDR/ADMM/CP';
try
    evalc("deblur('l2', 'XYZ', [], psf_std, b_std, FAST)");
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s  (no error was thrown)\n', tname);
catch e
    ok = all(cellfun(@(s) contains(e.message,s), {'PDR','ADMM','CP'}));
    if ok
        nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
    else
        nFail = nFail+1; failList{end+1} = tname;
        fprintf('  [FAIL]  %s\n         error msg: %s\n', tname, e.message);
    end
end

% Numeric algorithm string
tname = 'S4: numeric algorithm string throws error';
try
    evalc("deblur('l2', '123', [], psf_std, b_std, FAST)");
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s  (no error was thrown)\n', tname);
catch
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
end

% Empty algorithm string
tname = 'S4: empty algorithm string throws error';
try
    evalc("deblur('l2', '', [], psf_std, b_std, FAST)");
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s  (no error was thrown)\n', tname);
catch
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
end

% =============================================================================
fprintf('\n── S5: xinitial handling ────────────────────────────────────────────────\n');
% =============================================================================

% [] start → valid output
tname = 'S5: xinitial=[] gives valid output';
try
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Two calls with [] give identical results (deterministic)
tname = 'S5: repeated calls with xinitial=[] are deterministic';
try
    evalc("xa = deblur('l2','PDDR',[],psf_std,b_std,FAST)");
    evalc("xb = deblur('l2','PDDR',[],psf_std,b_std,FAST)");
    assert(isequal(xa, xb), 'results differ between identical calls');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% All-zero start
tname = 'S5: xinitial=zeros is accepted, output valid';
try
    x0z = zeros(H, W);
    evalc("x = deblur('l2','PDDR',x0z,psf_std,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% All-ones start
tname = 'S5: xinitial=ones is accepted, output valid';
try
    x0o = ones(H, W);
    evalc("x = deblur('l2','PDDR',x0o,psf_std,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Warm start from x_true
tname = 'S5: xinitial=x_true (warm start), output valid';
try
    evalc("x = deblur('l2','PDDR',x_true,psf_std,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Different xinitial → different result (confirms it's actually used)
tname = 'S5: different xinitial gives different result';
try
    i_1iter = struct('maxiter', 1, 'tol', 0);
    evalc("x_from_zero = deblur('l2','PDDR',zeros(H,W),psf_std,b_std,i_1iter)");
    evalc("x_from_ones = deblur('l2','PDDR',ones(H,W),psf_std,b_std,i_1iter)");
    assert(~isequal(x_from_zero, x_from_ones), 'xinitial has no effect on result');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% =============================================================================
fprintf('\n── S6: Parameter struct overrides ───────────────────────────────────────\n');
% =============================================================================

% maxiter=1 runs exactly one step
tname = 'S6: maxiter=1 gives valid output';
try
    i1 = struct('maxiter', 1, 'tol', 0);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,i1)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% More iterations → different result from fewer
tname = 'S6: maxiter=1 and maxiter=50 give different results';
try
    i1  = struct('maxiter',  1, 'tol', 0);
    i50 = struct('maxiter', 50, 'tol', 0);
    evalc("x1  = deblur('l2','PDDR',[],psf_std,b_std,i1)");
    evalc("x50 = deblur('l2','PDDR',[],psf_std,b_std,i50)");
    assert(~isequal(x1, x50), 'maxiter has no effect');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% gamma override changes output
tname = 'S6: different i.gamma gives different output';
try
    ig_lo = struct('maxiter', 20, 'tol', 0, 'gamma', 1e-5);
    ig_hi = struct('maxiter', 20, 'tol', 0, 'gamma', 1.0);
    evalc("x_lo = deblur('l2','PDDR',[],psf_std,b_std,ig_lo)");
    evalc("x_hi = deblur('l2','PDDR',[],psf_std,b_std,ig_hi)");
    assert(~isequal(x_lo, x_hi), 'i.gamma has no effect on result');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% gamma=0 (no TV term) — still valid
tname = 'S6: i.gamma=0 gives valid output (no TV)';
try
    ig0 = struct('maxiter', 5, 'tol', 0, 'gamma', 0);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,ig0)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% tol=1.0 → should converge on first iteration
tname = 'S6: tol=1.0 allows early convergence, output valid';
try
    itol = struct('maxiter', 500, 'tol', 1.0);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,itol)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% i.t override
tname = 'S6: i.t=0.5 override accepted (PDDR)';
try
    it = struct('maxiter', 5, 'tol', 0, 't', 0.5);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,it)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% i.rho override
tname = 'S6: i.rho=1.5 override accepted (ADMM)';
try
    irho = struct('maxiter', 5, 'tol', 0, 'rho', 1.5);
    evalc("x = deblur('l2','ADMM',[],psf_std,b_std,irho)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% i.s override (CP only)
tname = 'S6: i.s=0.1 override accepted (CP)';
try
    is = struct('maxiter', 5, 'tol', 0, 's', 0.1);
    evalc("x = deblur('l2','CP',[],psf_std,b_std,is)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Unknown field in i is silently ignored
tname = 'S6: unknown field in i is silently ignored';
try
    i_extra = struct('maxiter', 5, 'tol', 0, 'nonexistent_field', 999);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,i_extra)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% =============================================================================
fprintf('\n── S7: Default parameter correctness ───────────────────────────────────\n');
% =============================================================================

% l1 and l2 defaults use different gamma → different outputs
tname = 'S7: l1 and l2 defaults give different outputs (gamma is problem-dependent)';
try
    evalc("xl1 = deblur('l1','PDDR',[],psf_std,b_std,FAST)");
    evalc("xl2 = deblur('l2','PDDR',[],psf_std,b_std,FAST)");
    assert(~isequal(xl1, xl2), 'l1 and l2 give identical output — gamma may not be problem-dependent');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% PDR default t=0.25 confirmed by matching explicit override
tname = 'S7: PDR default t/rho matches expected 0.25/1.25';
try
    i_exp = struct('maxiter', 5, 'tol', 0, 't', 0.25, 'rho', 1.25);
    evalc("x_def = deblur('l2','PDR',[],psf_std,b_std,FAST)");
    evalc("x_exp = deblur('l2','PDR',[],psf_std,b_std,i_exp)");
    assert(isequal(x_def, x_exp), 'PDR default t/rho does not match 0.25/1.25');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% PDDR default t=1.0, rho=1.0 confirmed
tname = 'S7: PDDR default t/rho matches 1.0/1.0';
try
    i_exp = struct('maxiter', 5, 'tol', 0, 't', 1.0, 'rho', 1.0);
    evalc("x_def = deblur('l2','PDDR',[],psf_std,b_std,FAST)");
    evalc("x_exp = deblur('l2','PDDR',[],psf_std,b_std,i_exp)");
    assert(isequal(x_def, x_exp), 'PDDR default t/rho does not match 1.0/1.0');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% ADMM default maxiter=200 (not 500)
tname = 'S7: ADMM default maxiter=200 (not 500)';
try
    i_200 = struct('maxiter', 200, 'tol', 0);
    i_500 = struct('maxiter', 500, 'tol', 0);
    evalc("x_def = deblur('l2','ADMM',[],psf_std,b_std,struct('tol',0))");
    evalc("x_200 = deblur('l2','ADMM',[],psf_std,b_std,i_200)");
    evalc("x_500 = deblur('l2','ADMM',[],psf_std,b_std,i_500)");
    assert( isequal(x_def, x_200), 'ADMM default maxiter is not 200');
    assert(~isequal(x_def, x_500), 'ADMM default maxiter appears to be 500');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% =============================================================================
fprintf('\n── S8: Edge-case input images ───────────────────────────────────────────\n');
% =============================================================================

% All-zero image
tname = 'S8: b=zeros (all-zero input)';
try
    b_zeros = zeros(H, W);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_zeros,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% All-ones image
tname = 'S8: b=ones (all-ones input)';
try
    b_ones = ones(H, W);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_ones,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Constant-gray image
tname = 'S8: b=0.5*ones (uniform gray)';
try
    b_gray = 0.5 * ones(H, W);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_gray,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Image with boundary-touching values
tname = 'S8: b with exact 0 and 1 values (boundary touch)';
try
    b_bnd = b_std;
    b_bnd(1,1) = 0.0;
    b_bnd(1,2) = 1.0;
    evalc("x = deblur('l2','PDDR',[],psf_std,b_bnd,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Non-square image (height ≠ width)
tname = 'S8: non-square image 32×48 — output size matches input';
try
    H2 = 32; W2 = 48;
    b_ns = max(0, min(1, imfilter(rand(H2,W2), psf_std, 'circular') + 0.02*randn(H2,W2)));
    evalc("x = deblur('l2','PDDR',[],psf_std,b_ns,FAST)");
    assert(isequal(size(x), [H2, W2]),          'output size does not match non-square input');
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Non-square image — all four algorithms
for ai = 1:numel(ALGS)
    alg   = ALGS{ai};
    tname = sprintf('S8: non-square 32×48 + %s', alg);
    try
        H2 = 32; W2 = 48;
        b_ns = max(0, min(1, imfilter(rand(H2,W2), psf_std, 'circular') + 0.02*randn(H2,W2)));
        evalc("x = deblur('l2', alg, [], psf_std, b_ns, FAST)");
        assert(isequal(size(x), [H2, W2]));
        assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
        nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
    catch e
        nFail = nFail+1; failList{end+1} = tname;
        fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
    end
end

% =============================================================================
fprintf('\n── S9: Edge-case PSFs ───────────────────────────────────────────────────\n');
% =============================================================================

% Delta (identity) PSF — no blur applied
tname = 'S9: delta PSF (no blur) gives valid output';
try
    psf_delta = zeros(9,9); psf_delta(5,5) = 1;
    evalc("x = deblur('l2','PDDR',[],psf_delta,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% 1×1 PSF (scalar, acts as identity)
tname = 'S9: 1×1 scalar PSF';
try
    psf_1x1 = 1;
    evalc("x = deblur('l2','PDDR',[],psf_1x1,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Uniform (box) PSF
tname = 'S9: 5×5 uniform box PSF';
try
    psf_box = ones(5,5) / 25;
    evalc("x = deblur('l2','PDDR',[],psf_box,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Non-square PSF (3×7)
tname = 'S9: non-square 3×7 PSF';
try
    psf_asym = ones(3,7) / 21;
    evalc("x = deblur('l2','PDDR',[],psf_asym,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Motion PSF
tname = 'S9: motion PSF (len=9, angle=30)';
try
    psf_motion = motionPSF(9, 30);
    evalc("x = deblur('l2','PDDR',[],psf_motion,b_std,FAST)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% =============================================================================
fprintf('\n── S10: Small images ────────────────────────────────────────────────────\n');
% =============================================================================
% Use a delta PSF so PSF padding does not exceed image size.

for sz = [4, 8, 16]
    for ai = 1:numel(ALGS)
        alg   = ALGS{ai};
        tname = sprintf('S10: %d×%d image + %s', sz, sz, alg);
        try
            b_small   = max(0, min(1, 0.5 + 0.3*randn(sz, sz)));
            half      = floor(sz/2) + 1;
            psf_small = zeros(sz, sz); psf_small(half, half) = 1;
            evalc("x = deblur('l2', alg, [], psf_small, b_small, FAST)");
            assert(isequal(size(x), [sz, sz]), 'wrong size');
            assert(all(isfinite(x(:))),        'non-finite');
            assert(all(x(:)>=0 & x(:)<=1),    'outside [0,1]');
            nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
        catch e
            nFail = nFail+1; failList{end+1} = tname;
            fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
        end
    end
end

% =============================================================================
fprintf('\n── S11: Huber problem ───────────────────────────────────────────────────\n');
% =============================================================================

% All four algorithms must accept 'huber'
for ai = 1:numel(ALGS)
    alg   = ALGS{ai};
    tname = sprintf('S11: huber + %s gives valid output', alg);
    try
        ih = struct('maxiter', 5, 'tol', 0, 'delta', 0.1);
        evalc("x = deblur('huber', alg, [], psf_std, b_std, ih)");
        assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
        nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
    catch e
        nFail = nFail+1; failList{end+1} = tname;
        fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
    end
end

% i.delta changes output
tname = 'S11: different i.delta gives different output (PDDR)';
try
    ih_lo = struct('maxiter', 10, 'tol', 0, 'delta', 0.01);
    ih_hi = struct('maxiter', 10, 'tol', 0, 'delta', 1.0);
    evalc("xd1 = deblur('huber','PDDR',[],psf_std,b_std,ih_lo)");
    evalc("xd2 = deblur('huber','PDDR',[],psf_std,b_std,ih_hi)");
    assert(~isequal(xd1, xd2), 'i.delta has no effect on huber output');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% =============================================================================
fprintf('\n── S12: Extreme parameter values ────────────────────────────────────────\n');
% =============================================================================

% Very small t (near zero)
tname = 'S12: t=1e-6 gives valid output (PDDR)';
try
    i_st = struct('maxiter', 5, 'tol', 0, 't', 1e-6);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,i_st)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Large t (outside typical convergence range — boxProx must still clamp output)
tname = 'S12: t=100 output clamped to [0,1] by boxProx (PDDR)';
try
    i_bt = struct('maxiter', 5, 'tol', 0, 't', 100);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,i_bt)");
    assert(all(x(:)>=0 & x(:)<=1), 'boxProx failed to clamp large-t output');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% Very large gamma (heavy TV — image should approach constant)
tname = 'S12: gamma=1000 output clamped to [0,1] (heavy TV)';
try
    i_bg = struct('maxiter', 5, 'tol', 0, 'gamma', 1000);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,i_bg)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% rho just below 2 (upper convergence boundary)
tname = 'S12: rho=1.99 (near upper DR boundary) gives valid output';
try
    i_rh = struct('maxiter', 5, 'tol', 0, 'rho', 1.99);
    evalc("x = deblur('l2','PDDR',[],psf_std,b_std,i_rh)");
    assert(all(isfinite(x(:))) && all(x(:)>=0 & x(:)<=1));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% CP outside convergence region (t*s*||A||^2 > 1) — output must still be in [0,1]
tname = 'S12: CP with t=s=10 (outside convergence) — output clamped by boxProx';
try
    i_cpd = struct('maxiter', 5, 'tol', 0, 't', 10, 's', 10);
    evalc("x = deblur('l2','CP',[],psf_std,b_std,i_cpd)");
    assert(all(x(:)>=0 & x(:)<=1), 'boxProx failed to clamp CP divergent output');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% tol=0 forces full maxiter (no early stop), then verify we get maxiter iters
tname = 'S12: tol=0 forces full maxiter (no early termination)';
try
    % With tol=0, rel_change < tol is never true, so all maxiter steps run.
    % Result must differ from tol=1 (which exits on first step).
    i_tol0 = struct('maxiter', 30, 'tol', 0);
    i_tol1 = struct('maxiter', 30, 'tol', 1.0);
    evalc("x_full  = deblur('l2','PDDR',[],psf_std,b_std,i_tol0)");
    evalc("x_early = deblur('l2','PDDR',[],psf_std,b_std,i_tol1)");
    assert(~isequal(x_full, x_early), 'tol=0 and tol=1 give same result — early stop may not be working');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% =============================================================================
fprintf('\n── S13: Proximal operators (unit tests) ─────────────────────────────────\n');
% =============================================================================

% boxProx: clips to [0,1]
tname = 'S13: boxProx clips to [0,1]';
try
    v = [-1, -0.5, 0, 0.5, 1, 1.5, 2];
    out = boxProx(v);
    assert(all(out >= 0 & out <= 1),      'values outside [0,1]');
    assert(out(1) == 0 && out(end) == 1,  'extremes not clamped');
    assert(out(4) == 0.5,                 'interior value changed');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% boxProx: preserves size and type
tname = 'S13: boxProx preserves matrix size';
try
    M = randn(8, 12);
    out = boxProx(M);
    assert(isequal(size(out), size(M)));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% l1Prox: soft threshold
tname = 'S13: l1Prox zeroes values inside threshold';
try
    v   = [-2, -0.3, 0, 0.3, 2];
    out = l1Prox(v, 0.5);
    assert(out(2) == 0 && out(4) == 0,   'values inside threshold not zeroed');
    assert(abs(out(1) - (-1.5)) < 1e-12, 'negative side shifted incorrectly');
    assert(abs(out(5) -   1.5 ) < 1e-12, 'positive side shifted incorrectly');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% l1Prox: t=0 is identity
tname = 'S13: l1Prox with t=0 is identity';
try
    v = randn(1, 20);
    assert(isequal(l1Prox(v, 0), v));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% isoProx: zero input stays zero
tname = 'S13: isoProx of zeros gives zeros';
try
    [u, v] = isoProx(zeros(4,4), zeros(4,4), 0.5);
    assert(all(u(:)==0) && all(v(:)==0));
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% isoProx: large t shrinks to zero
tname = 'S13: isoProx with large t shrinks to zero';
try
    a = randn(4,4); b_mat = randn(4,4);
    [u, v] = isoProx(a, b_mat, 1e6);
    assert(all(u(:)==0) && all(v(:)==0), 'large t did not shrink to zero');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% l1ShiftProx: shift property
tname = 'S13: l1ShiftProx = b + l1Prox(y-b, t)';
try
    y  = randn(1, 10);
    b_v = randn(1, 10);
    t  = 0.3;
    expected = b_v + l1Prox(y - b_v, t);
    got      = l1ShiftProx(y, b_v, t);
    assert(max(abs(expected(:) - got(:))) < 1e-12);
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% l2shiftprox: shift property
tname = 'S13: l2shiftprox = b + l2prox(y-b, t)';
try
    y  = randn(1, 10);
    b_v = randn(1, 10);
    t  = 0.5;
    expected = b_v + l2prox(y - b_v, t);
    got      = l2shiftprox(y, b_v, t);
    assert(max(abs(expected(:) - got(:))) < 1e-12);
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% l2prox: scaling property
tname = 'S13: l2prox scales by 1/(2t+1)';
try
    x_v = randn(1, 10); t = 2.0;
    expected = x_v / (2*t + 1);
    got      = l2prox(x_v, t);
    assert(max(abs(expected(:) - got(:))) < 1e-12);
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% huberShiftProx: values near zero use L2 branch (linear shrink)
tname = 'S13: huberShiftProx L2 branch for small residuals';
try
    y_v = [0.01; -0.01];    % small residuals, well inside delta
    b_v = zeros(2,1);
    t   = 1.0; delta = 0.5;
    out = huberShiftProx(y_v, b_v, t, delta);
    expected = y_v / (1 + t);   % L2 branch: shrink by 1/(1+t)
    assert(max(abs(out - expected)) < 1e-12, 'L2 branch of huberShiftProx incorrect');
    nPass = nPass+1; fprintf('  [PASS]  %s\n', tname);
catch e
    nFail = nFail+1; failList{end+1} = tname;
    fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
end

% =============================================================================
fprintf('\n── S14: Reconstruction quality (PSNR improvement) ──────────────────────\n');
% =============================================================================
% With enough iterations, every algorithm must beat the blurred input.

psnr_b   = computePSNR(x_true, b_std);
psnr_bsp = computePSNR(x_true, b_sp);
fprintf('       Input PSNR (Gaussian/L2): %.2f dB\n', psnr_b);
fprintf('       Input PSNR (S&P/L1):      %.2f dB\n', psnr_bsp);

i_qual = struct('maxiter', 200, 'tol', 1e-4);
for pi = 1:numel(PROBLEMS)
    prob = PROBLEMS{pi};
    % Use the noise type matched to the fidelity term
    if strcmp(prob, 'l1')
        b_test  = b_sp;
        psnr_in = psnr_bsp;
    else
        b_test  = b_std;
        psnr_in = psnr_b;
    end
    for ai = 1:numel(ALGS)
        alg   = ALGS{ai};
        tname = sprintf('S14: %s + %s PSNR > input', prob, alg);
        try
            evalc("x = deblur(prob, alg, [], psf_std, b_test, i_qual)");
            psnr_x = computePSNR(x_true, x);
            assert(psnr_x > psnr_in, ...
                sprintf('PSNR %.2f dB ≤ input %.2f dB', psnr_x, psnr_in));
            nPass = nPass+1;
            fprintf('  [PASS]  %s  (%.2f dB)\n', tname, psnr_x);
        catch e
            nFail = nFail+1; failList{end+1} = tname;
            fprintf('  [FAIL]  %s\n         %s\n', tname, e.message);
        end
    end
end

% =============================================================================
%  Summary
% =============================================================================
fprintf('\n%s\n', repmat('═', 1, 72));
total = nPass + nFail;
fprintf('  Results: %d / %d passed', nPass, total);
if nFail == 0
    fprintf('  ✓  All tests passed.\n');
else
    fprintf('   (%d failed)\n', nFail);
    fprintf('  Failed tests:\n');
    for k = 1:numel(failList)
        fprintf('    [%2d]  %s\n', k, failList{k});
    end
end
fprintf('%s\n\n', repmat('═', 1, 72));
