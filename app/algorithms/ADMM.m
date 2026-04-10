function [x_sol, info] = ADMM(b, psf, config)
% ADMM  Alternating Direction Method of Multipliers for image deblurring.
%       Converted from Ian's Python implementation.
%
% Solves one of two problems depending on config.problem:
%
%   (L1)  min_x  ||Kx - b||_1   + delta_[0,1](x) + gamma * ||Dx||_iso
%   (L2)  min_x  ||Kx - b||_2^2 + delta_[0,1](x) + gamma * ||Dx||_iso
%
% ADMM VARIABLE SPLITTING:
%   The problem is split into TWO equality constraints:
%       x = u          (so we can handle box constraint on u separately)
%       Ax = y         (so we can handle data fidelity + TV on y separately)
%   where A = [K; D_x; D_y].
%
%   This gives 9 variables total:
%     Primal:  x (image), u (box copy), y1/y2/y3 (A-copy, 3 channels)
%     Dual:    w (dual for x=u), z1/z2/z3 (dual for Ax=y, 3 channels)
%
% EACH ITERATION:
%   x-update:  Solve the normal equation  (I + A^T A) x = rhs  via FFT
%   u-update:  prox of box indicator  = proj_[0,1](...)
%   y-update:  prox of data fidelity + TV  (channel-wise)
%   w-update:  scaled dual update for x=u constraint
%   z-update:  scaled dual update for Ax=y constraint
%
% CONVERGENCE:
%   Three residuals are tracked:
%     primal_res_u  = ||x - u||       (primal feasibility for x=u)
%     primal_res_y  = ||Ax - y||      (primal feasibility for Ax=y)
%     step_norm     = ||x_new - x_old||  (how much x moved)
%   Converged when all three are below tol.
%
% USAGE:
%   config.problem = 'l2';    % or 'l1'
%   config.gamma   = 0.006;
%   config.t       = 1.0;     % ADMM penalty parameter
%   config.rho     = 1.0;     % over-relaxation (0 < rho < 2)
%   config.maxiter = 200;
%   config.tol     = 1e-4;
%   config.verbose = true;
%   [x_sol, info] = ADMM(b, psf, config);
%
% NOTE: The prox_operators/ folder must be on the MATLAB path.
%   addpath('../prox_operators')
%
% INPUTS:
%   b      - (H x W) double, blurred+noisy image, values in [0,1]
%   psf    - (kH x kW) double, point-spread function (blur kernel)
%   config - struct with optional fields (defaults shown above)
%
% OUTPUTS:
%   x_sol - (H x W) double, reconstructed image
%   info  - struct: .objective, .primal_res_u, .primal_res_y,
%                   .step_norm, .iterations, .converged, .time_sec

% -------------------------------------------------------------------------
% 1. Read config with defaults
% -------------------------------------------------------------------------
problem = fieldOrDefault(config, 'problem', 'l2');   % 'l1' or 'l2'
gamma   = fieldOrDefault(config, 'gamma',   0.006);  % TV regularization weight
t       = fieldOrDefault(config, 't',       1.0);    % ADMM penalty parameter
rho     = fieldOrDefault(config, 'rho',     1.0);    % over-relaxation (0 < rho < 2)
maxiter = fieldOrDefault(config, 'maxiter', 200);    % max iterations
tol     = fieldOrDefault(config, 'tol',     1e-4);   % convergence tolerance
verbose = fieldOrDefault(config, 'verbose', true);   % print progress?

% -------------------------------------------------------------------------
% 2. Pre-compute fixed quantities
% -------------------------------------------------------------------------
[H, W] = size(b);

% Fourier-domain representation of the blur operator K
k_hat = psfToOtf(psf, H, W);   % (H x W) complex OTF

% Denominator for the x-update linear solve  (I + A^T A)^{-1}
%   A^T A = K^T K + D^T D  (blur^T blur + Laplacian)
%   In Fourier: |k_hat|^2 + laplacianSymbol
% Note: same denominator as PDR (no t^2 factor); t appears in the RHS instead.
denom = 1.0 + abs(k_hat).^2 + laplacianSymbol(H, W);   % (H x W) real

% -------------------------------------------------------------------------
% 3. Initialize all 9 ADMM variables
% -------------------------------------------------------------------------
x = b;              % primal image variable (H x W), start from observation
if isfield(config, 'x_init') && ~isempty(config.x_init)
    x = config.x_init;   % use caller-supplied initial guess
end

u = b;              % auxiliary copy of x for the box constraint (H x W)
w = zeros(H, W);    % scaled dual for the x = u constraint (H x W)

% Initialize y = Ax (the auxiliary copy of Ax for data fidelity + TV)
[y1, y2, y3] = applyA(x, k_hat);   % y1: blur output, y2: horiz grad, y3: vert grad

z1 = zeros(H, W);   % scaled dual for Kx = y1 constraint
z2 = zeros(H, W);   % scaled dual for D_x x = y2 constraint
z3 = zeros(H, W);   % scaled dual for D_y x = y3 constraint

% Preallocate history arrays
obj_history        = zeros(maxiter, 1);  % objective value per iteration
primal_res_u_hist  = zeros(maxiter, 1);  % ||x - u|| per iteration
primal_res_y_hist  = zeros(maxiter, 1);  % ||Ax - y|| per iteration
step_norm_hist     = zeros(maxiter, 1);  % ||x_new - x_old|| per iteration

converged = false;

if verbose
    fprintf('Starting ADMM\n');
    fprintf('problem=%s, gamma=%.4f, t=%.2f, rho=%.2f, maxiter=%d\n', ...
            problem, gamma, t, rho, maxiter);
end

t_start = tic;

% -------------------------------------------------------------------------
% 4. Main ADMM iteration
% -------------------------------------------------------------------------
for k = 1:maxiter

    x_old = x;      % save previous x to measure step size

    % =================================================================
    % X-UPDATE
    %   Solve the linear system  (I + A^T A) x = rhs
    %   where rhs comes from the dual-augmented Lagrangian optimality.
    %
    %   Full derivation: the ADMM x-step minimizes over x:
    %     ||x - u||^2/2 + ||Ax - y||^2/2 + <w, x-u> + <z, Ax-y>
    %   (in scaled dual form, w and z are the dual variables divided by t)
    %   Taking gradient = 0 gives the normal equations.
    % =================================================================

    % Compute A^T y (adjoint of A applied to current y)
    ATy = applyAT(y1, y2, y3, k_hat);  % (H x W)

    % Compute A^T z (adjoint of A applied to dual variables z)
    ATz = applyAT(z1, z2, z3, k_hat);  % (H x W)

    % Right-hand side: u + A^T y - (1/t) * (w + A^T z)
    % The 1/t comes from the scaled dual form of ADMM
    rhs = u + ATy - (1.0/t) * (w + ATz);   % (H x W)

    % Solve  (I + A^T A) x = rhs  via FFT (diagonal in Fourier domain)
    x = real(ifft2(fft2(rhs) ./ denom));   % (H x W)

    % =================================================================
    % U-UPDATE
    %   u is the auxiliary copy of x for the box constraint.
    %   Its update is:  prox_{(1/t) * delta_[0,1]}(argument)
    %   Since t > 0, the prox of an indicator is always projection (no t).
    %
    %   The argument uses over-relaxation (rho):
    %     rho * x + (1-rho) * u_old + w/t
    %   When rho=1 this is just  x + w/t  (standard ADMM).
    %   rho in (1, 2) gives over-relaxation, which can speed convergence.
    % =================================================================
    u_arg = rho * x + (1.0 - rho) * u + w / t;     % over-relaxed argument
    u     = boxProx(u_arg);                          % project onto [0,1]

    % Optional live display callback (used by the app for per-iteration display)
    if isfield(config, 'display_callback')
        config.display_callback(u, k);
    end

    % =================================================================
    % Y-UPDATE
    %   y1/y2/y3 are the auxiliary copies of Ax for data fidelity + TV.
    %   Update y by applying prox of g = (data_fidelity + gamma*TV).
    %   g is separable across channels, so we update each independently.
    %
    %   The argument again uses over-relaxation:
    %     rho * A x + (1-rho) * y_old + z/t
    % =================================================================

    % Compute current A*x (forward operator applied to new x)
    [Ax1, Ax2, Ax3] = applyA(x, k_hat);

    % Form over-relaxed arguments for each channel
    y1_arg = rho * Ax1 + (1.0 - rho) * y1 + z1 / t;
    y2_arg = rho * Ax2 + (1.0 - rho) * y2 + z2 / t;
    y3_arg = rho * Ax3 + (1.0 - rho) * y3 + z3 / t;

    % tau = 1/t is the effective step size for the prox of g
    tau = 1.0 / t;

    % Channel 1: data fidelity prox  = prox_{tau * ||. - b||} (y1_arg)
    if strcmp(problem, 'l1')
        y1 = l1ShiftProx(y1_arg, b, tau);   % soft-threshold shifted by b
    else
        y1 = l2shiftprox(y1_arg, b, tau);   % quadratic shrinkage toward b
    end

    % Channels 2 & 3: isotropic TV prox  = prox_{tau * gamma * iso}(y2, y3)
    [y2, y3] = isoProx(y2_arg, y3_arg, gamma * tau);

    % =================================================================
    % DUAL UPDATES (w and z)
    %   Standard ADMM dual step: dual += t * (primal residual)
    %   w tracks the x=u constraint violation
    %   z1/z2/z3 track the Ax=y constraint violation
    % =================================================================
    w  = w  + t * (x  - u);    % dual update for x = u constraint
    z1 = z1 + t * (Ax1 - y1); % dual update for Kx = y1 constraint
    z2 = z2 + t * (Ax2 - y2); % dual update for D_x x = y2 constraint
    z3 = z3 + t * (Ax3 - y3); % dual update for D_y x = y3 constraint

    % =================================================================
    % Convergence diagnostics -- three residuals (Ian's stopping rule)
    % =================================================================

    % How well is x = u satisfied? (primal feasibility for box constraint)
    primal_res_u = norm(x(:) - u(:));

    % How well is Ax = y satisfied? (primal feasibility for A constraint)
    % Sum of squared norms across all three channels, then square root
    primal_res_y = sqrt(norm(Ax1(:) - y1(:))^2 + ...
                        norm(Ax2(:) - y2(:))^2 + ...
                        norm(Ax3(:) - y3(:))^2);

    % How much did x move this iteration?
    step_norm = norm(x(:) - x_old(:));

    % Store residuals for output
    primal_res_u_hist(k) = primal_res_u;
    primal_res_y_hist(k) = primal_res_y;
    step_norm_hist(k)    = step_norm;

    % Compute objective value at current x
    Kx = applyBlur(x, k_hat);
    [gx, gy] = imageGrad(x);
    tv_val = sum(sum(sqrt(gx.^2 + gy.^2)));    % isotropic TV

    if strcmp(problem, 'l1')
        fidelity = sum(abs(Kx(:) - b(:)));
    else
        fidelity = sum((Kx(:) - b(:)).^2);
    end
    obj_history(k) = fidelity + gamma * tv_val;

    % Print progress every 10 iterations
    if verbose && (k == 1 || mod(k, 10) == 0 || k == maxiter)
        fprintf('  iter %4d | obj=%.6e | ||x-u||=%.3e | ||Ax-y||=%.3e | step=%.3e\n', ...
                k, obj_history(k), primal_res_u, primal_res_y, step_norm);
    end

    % Stop when ALL three residuals fall below tolerance
    if max([primal_res_u, primal_res_y, step_norm]) < tol
        converged = true;
        if verbose
            fprintf('  Converged at iteration %d.\n', k);
        end
        break   % exit for loop
    end

end  % end for k

% -------------------------------------------------------------------------
% 5. Package outputs
% -------------------------------------------------------------------------
x_sol = boxProx(x);     % final solution (project onto [0,1] for safety)

elapsed = toc(t_start);

if verbose
    fprintf('\nADMM Results\n');
    fprintf('  status    : %s\n', iif(converged, 'converged', 'maxiter reached'));
    fprintf('  iters     : %d\n', k);
    fprintf('  objective : %.6e\n', obj_history(k));
    fprintf('  time (s)  : %.3f\n', elapsed);
end

info.objective     = obj_history(1:k);
info.primal_res_u  = primal_res_u_hist(1:k);
info.primal_res_y  = primal_res_y_hist(1:k);
info.step_norm     = step_norm_hist(1:k);
info.iterations    = k;
info.converged     = converged;
info.time_sec      = elapsed;

end  % <<<< end of ADMM (main function)


% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function k_hat = psfToOtf(psf, H, W)
% Pad PSF to image size, shift center to (1,1), then FFT.
[kH, kW] = size(psf);
padded = zeros(H, W);
padded(1:kH, 1:kW) = psf;      % PSF in top-left corner
padded = circshift(padded, -floor(kH/2), 1);    % shift center row to row 1
padded = circshift(padded, -floor(kW/2), 2);    % shift center col to col 1
k_hat  = fft2(padded);          % OTF = eigenvalues of K
end


function out = applyBlur(x, k_hat)
% Kx = IFFT(k_hat .* FFT(x))
out = real(ifft2(k_hat .* fft2(x)));
end


function out = applyBlurAdj(x, k_hat)
% K^T x = IFFT(conj(k_hat) .* FFT(x))
out = real(ifft2(conj(k_hat) .* fft2(x)));
end


function [gx, gy] = imageGrad(x)
% Forward finite differences with circular (periodic) boundary.
gx = circshift(x, -1, 2) - x;  % horizontal: x(i,j+1) - x(i,j)
gy = circshift(x, -1, 1) - x;  % vertical:   x(i+1,j) - x(i,j)
end


function out = imageDivAdj(gx, gy)
% Adjoint of imageGrad (backward differences, periodic boundary).
out = (circshift(gx, 1, 2) - gx) + (circshift(gy, 1, 1) - gy);
end


function [Ax1, Ax2, Ax3] = applyA(x, k_hat)
% A = [K; D_x; D_y]: blur stacked with gradient
Ax1 = applyBlur(x, k_hat);         % channel 1: Kx (blurred)
[Ax2, Ax3] = imageGrad(x);         % channels 2 & 3: gradient
end


function out = applyAT(y1, y2, y3, k_hat)
% A^T [y1; y2; y3] = K^T y1 + div(y2, y3)
out = applyBlurAdj(y1, k_hat) + imageDivAdj(y2, y3);
end


function D = laplacianSymbol(H, W)
% Fourier eigenvalues of the discrete Laplacian D^T D (periodic BC).
wx = (2*pi/W) * (0:W-1);       % horizontal frequencies (1 x W)
wy = (2*pi/H) * (0:H-1)';      % vertical frequencies   (H x 1)
D  = (2 - 2*cos(wx)) + (2 - 2*cos(wy));    % broadcast -> (H x W)
end


function val = fieldOrDefault(s, field, default)
% Return s.field if it exists, else return default.
if isfield(s, field)
    val = s.(field);
else
    val = default;
end
end


function out = iif(cond, a, b)
% Inline if: return a if cond is true, else b.
% (MATLAB doesn't have a ternary operator, so this is a common workaround.)
if cond
    out = a;
else
    out = b;
end
end


