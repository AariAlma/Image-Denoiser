function [x_sol, info] = CP(b, psf, config)
% CP  Chambolle-Pock (Primal-Dual Hybrid Gradient) solver for image deblurring.
%     Based on David's implementation.
%
% Solves one of two problems depending on config.problem:
%
%   (L1)  min_x  ||Kx - b||_1         + delta_[0,1](x) + gamma * ||Dx||_iso
%   (L2)  min_x  0.5 * ||Kx - b||_2^2 + delta_[0,1](x) + gamma * ||Dx||_iso
%
% CHAMBOLLE-POCK (Algorithm 4 / PDHG):
%   The problem is written as  min_x f(x) + g(Ax)  with A = [K; D_x; D_y].
%   CP alternates between:
%     - A DUAL gradient ascent step on y  (ascend in g* direction)
%     - A PRIMAL proximal step on x       (descend in f direction)
%     - An EXTRAPOLATION step on z        (over-relaxed x for the next dual step)
%
%   This is different from PDDR/PDR/ADMM because:
%     - There is NO linear system solve (no FFT inversion per iteration)
%     - Instead there is explicit gradient ascent on the dual, then a prox step
%     - It requires TWO step sizes: t (primal) and s (dual), with t*s <= 1/||A||^2
%     - The prox of the primal (f = delta_[0,1]) is just a box projection
%
% EACH ITERATION:
%   y_new = prox_{s * g*}(y + s * A z)     -- dual ascent (Moreau for g*)
%   x_new = prox_{t * f}(x - t * A^T y)    -- primal descent (box projection)
%   z_new = 2 * x_new - x                  -- extrapolation (Polyak momentum)
%
% STEP SIZE RULE:
%   For convergence we need  t * s * ||A||^2 <= 1.
%   A safe choice: t = s = 1 / sqrt(||K||^2 + 8)
%   (||K||^2 <= 1 for normalized blur; ||D||^2 <= 8 for 2D gradient)
%
% USAGE:
%   config.problem = 'l2';    % or 'l1'
%   config.gamma   = 0.006;
%   config.t       = 0.2;     % primal step size
%   config.s       = 0.2;     % dual step size  (t*s <= 1/||A||^2)
%   config.maxiter = 500;
%   config.tol     = 1e-4;
%   config.verbose = true;
%   [x_sol, info] = CP(b, psf, config);
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
%   info  - struct: .objective, .rel_change, .iterations, .converged, .time_sec

% -------------------------------------------------------------------------
% 1. Read config with defaults
% -------------------------------------------------------------------------
problem = fieldOrDefault(config, 'problem', 'l2');   % 'l1' or 'l2'
gamma   = fieldOrDefault(config, 'gamma',   0.006);  % TV regularization weight
t       = fieldOrDefault(config, 't',       0.2);    % primal step size (tau)
s       = fieldOrDefault(config, 's',       0.2);    % dual step size (sigma)
maxiter = fieldOrDefault(config, 'maxiter', 500);    % max iterations
tol     = fieldOrDefault(config, 'tol',     1e-4);   % convergence tolerance
verbose = fieldOrDefault(config, 'verbose', true);   % print progress?

% -------------------------------------------------------------------------
% 2. Pre-compute fixed quantities
% -------------------------------------------------------------------------
[H, W] = size(b);

% OTF: Fourier-domain representation of blur operator K
k_hat = psfToOtf(psf, H, W);   % (H x W) complex

% -------------------------------------------------------------------------
% 3. Initialize variables
% -------------------------------------------------------------------------

% Primal variable: the image we are reconstructing
x = b;              % (H x W) start from the blurred observation

% Extrapolated primal: used in the dual update (starts equal to x)
z = b;              % (H x W) z = 2*x - x_prev (Polyak-like extrapolation)

% Dual variables: one per output channel of A = [K; D_x; D_y]
y1 = zeros(H, W);   % dual for blur channel (K x)
y2 = zeros(H, W);   % dual for horizontal gradient (D_x x)
y3 = zeros(H, W);   % dual for vertical gradient (D_y x)

obj_history = zeros(maxiter, 1);    % objective per iteration
rel_history = zeros(maxiter, 1);    % relative change per iteration

converged = false;
x_prev    = b;      % previous x for convergence check

t_start = tic;

% -------------------------------------------------------------------------
% 4. Main Chambolle-Pock iteration
% -------------------------------------------------------------------------
for k = 1:maxiter

    % =================================================================
    % DUAL UPDATE (y-step):
    %   y_new = prox_{s * g*}(y + s * A z)
    %
    %   We use the Moreau identity to evaluate prox_{s * g*}:
    %     prox_{s * g*}(v) = v - s * prox_{g/s}(v / s)
    %
    %   g is separable, so each channel is handled independently.
    %   We first compute v = y + s * A z (gradient ascent step in dual).
    % =================================================================

    % Forward operator applied to the extrapolated primal z
    [Az1, Az2, Az3] = applyA(z, k_hat);    % Az = [Kz; D_x z; D_y z]

    % Gradient ascent in the dual: v = y + s * A z
    v1 = y1 + s * Az1;     % blur channel
    v2 = y2 + s * Az2;     % horiz grad channel
    v3 = y3 + s * Az3;     % vert grad channel

    % Now apply prox_{s * g*} via Moreau identity:
    %   prox_{s*g*}(v) = v - s * prox_{g/s}(v/s)

    % Channel 1: data fidelity
    % g_1(y) = ||y - b||  ->  prox_{g_1/s}(v1/s) = shifted prox
    if strcmp(problem, 'l1')
        % L1 data fidelity
        % prox_{(1/s)*||.-b||_1}(v1/s)  = b + softThresh(v1/s - b, 1/s)
        p1 = l1ShiftProx(v1/s, b, 1/s);
    else
        % L2 data fidelity: g_1(y) = 0.5 * ||y - b||_2^2
        % prox_{(1/(2s))*||.-b||_2^2}(v1/s)  -- note the 0.5 factor!
        % prox_{tau*0.5*||.-b||^2}(v) = (v + tau*b) / (1 + tau)
        % here tau = 1/s (and we halve the penalty due to the 0.5)
        p1 = (v1 + b) / (s + 1);   % closed form for 0.5*||.-b||^2 case
    end
    % Moreau: y_new = v - s * p1
    y1 = v1 - s * p1;      % dual update for blur channel

    % Channels 2 & 3: isotropic TV  g_2(y2,y3) = gamma * iso(y2,y3)
    % prox_{(gamma/s)*iso}(v2/s, v3/s)
    [p2, p3] = isoProx(v2/s, v3/s, gamma/s);
    y2 = v2 - s * p2;      % dual update for horiz grad channel
    y3 = v3 - s * p3;      % dual update for vert grad channel

    % =================================================================
    % PRIMAL UPDATE (x-step):
    %   x_new = prox_{t * f}(x - t * A^T y)
    %   f = delta_[0,1], so prox_{t*f} = proj_[0,1] (no dependence on t)
    %
    %   This is a gradient descent step in the primal, followed by
    %   projecting back onto the feasible set [0,1].
    % =================================================================

    % Save old x for extrapolation
    x_prev_step = x;        % x before this update

    % Adjoint operator: A^T y = K^T y1 + div(y2, y3)
    Aty = applyAT(y1, y2, y3, k_hat);

    % Gradient descent + projection
    x = boxProx(x - t * Aty);  % project onto [0,1] after the gradient step

    % Optional live display callback (used by the app for per-iteration display)
    if isfield(config, 'display_callback')
        config.display_callback(x, k);
    end

    % =================================================================
    % EXTRAPOLATION (z-step):
    %   z = 2 * x_new - x_old
    %   This is like a momentum step: z overshoots x in the direction
    %   of the most recent change. It helps the dual update "look ahead"
    %   and accelerates convergence.
    % =================================================================
    z = 2 * x - x_prev_step;   % Polyak-like extrapolation

    % =================================================================
    % Convergence diagnostics
    % =================================================================

    % Relative change in the primal iterate
    rel_change = norm(x(:) - x_prev(:)) / (norm(x_prev(:)) + 1e-12);
    rel_history(k) = rel_change;

    % Compute objective value at current x
    [Kx, ~, ~] = applyA(x, k_hat);             % Kx (only need blur for objective)
    [gx, gy]   = imageGrad(x);                  % gradient for TV
    tv_val = sum(sum(sqrt(gx.^2 + gy.^2)));     % isotropic TV norm

    if strcmp(problem, 'l1')
        fidelity = sum(abs(Kx(:) - b(:)));          % L1 fidelity
    else
        fidelity = 0.5 * sum((Kx(:) - b(:)).^2);   % L2 fidelity (0.5 factor matches CP)
    end
    obj_history(k) = fidelity + gamma * tv_val;

    % Print progress every 50 iterations
    if verbose && (k == 1 || mod(k, 50) == 0)
        fprintf('  iter %4d  obj=%.6e  rel_change=%.6e\n', ...
                k, obj_history(k), rel_change);
    end

    % Stop if iterate barely changed (and at least 2 iters done)
    if rel_change < tol && k > 1
        converged = true;
        if verbose
            fprintf('  Converged at iteration %d  (rel_change=%.2e)\n', k, rel_change);
        end
        break   % exit for loop early
    end

    x_prev = x;     % update previous iterate for next convergence check
end  % end for k

% -------------------------------------------------------------------------
% 5. Package outputs
% -------------------------------------------------------------------------
x_sol = x;      % final solution (already in [0,1] from boxProx in last step)

elapsed = toc(t_start);

if verbose
    fprintf('  Done -- %d iters, %.2fs, converged=%d\n', k, elapsed, converged);
end

info.objective  = obj_history(1:k);
info.rel_change = rel_history(1:k);
info.iterations = k;
info.converged  = converged;
info.time_sec   = elapsed;

end  % <<<< end of CP (main function)


% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function k_hat = psfToOtf(psf, H, W)
% Pad PSF to image size, shift center to (1,1), then FFT.
[kH, kW] = size(psf);
padded = zeros(H, W);
padded(1:kH, 1:kW) = psf;
padded = circshift(padded, -floor(kH/2), 1);
padded = circshift(padded, -floor(kW/2), 2);
k_hat  = fft2(padded);
end


function out = applyBlur(x, k_hat)
% Kx = IFFT(k_hat .* FFT(x)) -- circular convolution via FFT
out = real(ifft2(k_hat .* fft2(x)));
end


function out = applyBlurAdj(x, k_hat)
% K^T x = IFFT(conj(k_hat) .* FFT(x)) -- adjoint of circular convolution
out = real(ifft2(conj(k_hat) .* fft2(x)));
end


function [gx, gy] = imageGrad(x)
% Forward finite differences (circular boundary).
gx = circshift(x, -1, 2) - x;  % horizontal: x(i, j+1) - x(i, j)
gy = circshift(x, -1, 1) - x;  % vertical:   x(i+1, j) - x(i, j)
end


function out = imageDivAdj(gx, gy)
% Adjoint of imageGrad (backward differences, circular boundary).
out = (circshift(gx, 1, 2) - gx) + (circshift(gy, 1, 1) - gy);
end


function [Ax1, Ax2, Ax3] = applyA(x, k_hat)
% A = [K; D_x; D_y]: blur + gradient
Ax1 = applyBlur(x, k_hat);
[Ax2, Ax3] = imageGrad(x);
end


function out = applyAT(y1, y2, y3, k_hat)
% A^T [y1; y2; y3] = K^T y1 + div(y2, y3)
out = applyBlurAdj(y1, k_hat) + imageDivAdj(y2, y3);
end


function val = fieldOrDefault(s, field, default)
% Return s.field if it exists, else return default.
if isfield(s, field)
    val = s.(field);
else
    val = default;
end
end
