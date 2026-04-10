function [x_sol, info] = PDDR(b, psf, config)
% PDDR  Primal-Dual Douglas-Rachford solver for image deblurring.
%
% Solves one of two problems depending on config.problem:
%
%   (L1)  min_x  ||Kx - b||_1   + delta_[0,1](x) + gamma * ||Dx||_iso
%   (L2)  min_x  ||Kx - b||_2^2 + delta_[0,1](x) + gamma * ||Dx||_iso
%
% where:
%   K           = blur operator (FFT-based, periodic boundary)
%   b           = blurred + noisy observation
%   delta_[0,1] = indicator of the box constraint (pixel values in [0,1])
%   D           = discrete gradient operator (D_x stacked with D_y)
%   ||Dx||_iso  = isotropic total variation seminorm
%   gamma       = regularization weight (controls smoothness vs data fit)
%
% The problem is rewritten as:  min_x  f(x) + g(Ax)   where
%   f(x)        = delta_[0,1](x)            -- primal cost
%   g(y1,y2,y3) = data_fit(y1) + gamma*iso(y2,y3)  -- dual cost (separable)
%   A           = [K; D_x; D_y]             -- stacked linear operator
%
% Douglas-Rachford splits the optimality condition into two resolvents:
%   M = (df, dg*)   -- resolved via prox operators (cheap, element-wise)
%   N = (A^T, -A)   -- resolved via FFT linear solve (fast, Fourier domain)
%
% USAGE:
%   config.problem = 'l2';    % or 'l1'
%   config.gamma   = 0.006;
%   config.t       = 1.0;
%   config.rho     = 1.0;
%   config.maxiter = 500;
%   config.tol     = 1e-4;
%   config.verbose = true;
%   [x_sol, info] = PDDR(b, psf, config);
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
% 1. Read config fields, falling back to sensible defaults
% -------------------------------------------------------------------------
problem = fieldOrDefault(config, 'problem', 'l2');   % which data fidelity term
gamma   = fieldOrDefault(config, 'gamma',   0.006);  % TV regularization weight
t       = fieldOrDefault(config, 't',       1.0);    % primal-dual step size
rho     = fieldOrDefault(config, 'rho',     1.0);    % DR relaxation (0 < rho < 2)
maxiter = fieldOrDefault(config, 'maxiter', 500);    % iteration cap
tol     = fieldOrDefault(config, 'tol',     1e-4);   % convergence tolerance
verbose = fieldOrDefault(config, 'verbose', true);   % whether to print progress

% -------------------------------------------------------------------------
% 2. Pre-compute fixed quantities (done once before the loop)
% -------------------------------------------------------------------------
[H, W] = size(b);       % H = number of rows, W = number of columns

% Convert PSF -> OTF (optical transfer function).
% The OTF k_hat is the array of eigenvalues of K in the Fourier domain.
% Once we have k_hat, applying K is just element-wise multiply then IFFT.
k_hat = psfToOtf(psf, H, W);   % (H x W) complex array

% Precompute the Fourier-domain denominator for the Step 3 linear solve.
% We need to invert  (I + t^2 * A^T A)  where A = [K; D_x; D_y].
%   A^T A = K^T K + D_x^T D_x + D_y^T D_y
%         = K^T K + discrete Laplacian
% Both are diagonalized by the DFT, so the inverse is just element-wise division.
%   Eigenvalues of K^T K    = |k_hat|^2
%   Eigenvalues of Laplacian = laplacianSymbol(H, W)
denom = 1.0 + t^2 * (abs(k_hat).^2 + laplacianSymbol(H, W));   % (H x W) real

% -------------------------------------------------------------------------
% 3. Initialize all variables
% -------------------------------------------------------------------------

% z is the primal "envelope" variable maintained by DR.
% It is NOT the solution itself -- the solution is proj_[0,1](z).
% Starting from b is a reasonable warm start.
z = b;                      % (H x W) primal envelope

% w1, w2, w3 are the dual envelope variables for each output channel of A:
%   w1 <--> blur channel   (K x, dimension H x W)
%   w2 <--> x-gradient     (D_x x, dimension H x W)
%   w3 <--> y-gradient     (D_y x, dimension H x W)
w1 = zeros(H, W);           % dual envelope for blur
w2 = zeros(H, W);           % dual envelope for horizontal gradient
w3 = zeros(H, W);           % dual envelope for vertical gradient

% Preallocate history arrays (faster than growing inside the loop)
obj_history = zeros(maxiter, 1);    % objective value at each iteration
rel_history = zeros(maxiter, 1);    % relative iterate change at each iteration

converged = false;          % will become true if we hit the tolerance
x_prev    = b;              % previous iterate for measuring change

t_start = tic;              % start the wall-clock timer (tic/toc like a stopwatch)

% -------------------------------------------------------------------------
% 4. Main Douglas-Rachford iteration
% -------------------------------------------------------------------------
for k = 1:maxiter

    % =================================================================
    % STEP 1a -- Resolvent of M, primal block
    %   Compute  x_half = prox_{t*f}(z)
    %   f = delta_[0,1], so its prox is the projection onto [0,1]
    % =================================================================
    x_half = boxProx(z);    % clip each pixel of z to [0,1]

    % =================================================================
    % STEP 1b -- Resolvent of M, dual block
    %   Compute  v_half = prox_{t*g*}(w)  via the Moreau identity:
    %     prox_{t*g*}(w) = w - t * prox_{g/t}(w/t)
    %
    %   The dual cost g is separable across channels, so we handle each
    %   channel independently.
    % =================================================================

    % --- Blur channel (data fidelity) ---
    % g_1(y) = ||y - b||  (either L1 or L2 norm)
    % prox_{(1/t)*g_1}(w1/t) = l1/l2 shifted proximal operator
    if strcmp(problem, 'l1')
        % L1 data fidelity: soft-threshold shifted by b
        pg1 = l1ShiftProx(w1 / t, b, 1/t);     % prox_{(1/t)*||.-b||_1}(w1/t)
    else
        % L2 data fidelity: shrink toward b
        pg1 = l2shiftprox(w1 / t, b, 1/t);     % prox_{(1/t)*||.-b||_2^2}(w1/t)
    end
    v1_half = w1 - t * pg1;     % Moreau: w1 - t * prox_{g1/t}(w1/t)

    % --- TV channel (isotropic total variation) ---
    % g_2(y2, y3) = gamma * ||(y2, y3)||_iso  (vector norm of gradient)
    % prox_{(gamma/t)*iso}(w2/t, w3/t)
    [pg2, pg3] = isoProx(w2 / t, w3 / t, gamma / t);
    v2_half = w2 - t * pg2;     % Moreau for x-gradient dual
    v3_half = w3 - t * pg3;     % Moreau for y-gradient dual

    % =================================================================
    % STEP 2 -- Douglas-Rachford reflection of M
    %   The DR update uses  2 * resolvent(M) - identity
    %   This "reflects" each variable through its half-step value.
    % =================================================================
    p  = 2*x_half  - z;     % reflect primal: 2*x_half - z
    q1 = 2*v1_half - w1;    % reflect blur dual
    q2 = 2*v2_half - w2;    % reflect x-gradient dual
    q3 = 2*v3_half - w3;    % reflect y-gradient dual

    % =================================================================
    % STEP 3 -- Resolvent of t*N (the linear/Fourier step)
    %   Solve:  (I + t^2 * A^T A) x_new = p - t * A^T q
    %
    %   Because A^T A is diagonalized by the DFT, this is:
    %     X_new(freq) = FFT(rhs)(freq) / denom(freq)
    %   then x_new = IFFT(X_new), which is fast.
    % =================================================================

    % Compute A^T q = K^T q1 + div(q2, q3)
    ATq = applyAT(q1, q2, q3, k_hat);  % (H x W)

    % Form the right-hand side of the linear system
    rhs = p - t * ATq;      % (H x W)

    % Solve in Fourier domain: element-wise divide by denom
    % fft2 transforms to Fourier, ./ divides element-wise, ifft2 brings back
    % real() discards tiny numerical imaginary parts (result should be real)
    x_new = real(ifft2(fft2(rhs) ./ denom));   % (H x W)

    % Compute the dual update: y_new = q + t * A x_new
    [Ax1, Ax2, Ax3] = applyA(x_new, k_hat);    % split A x_new into 3 channels
    y1_new = q1 + t * Ax1;     % blur channel dual update
    y2_new = q2 + t * Ax2;     % x-gradient dual update
    y3_new = q3 + t * Ax3;     % y-gradient dual update

    % =================================================================
    % STEP 4 -- Relaxed update (DR averaging)
    %   Move each envelope variable toward its new value by step rho.
    %   rho = 1 is the standard update; rho in (0,2) gives convergence.
    %   Using rho = 1.8 often speeds up convergence in practice.
    % =================================================================
    z  = z  + rho * (x_new  - x_half);     % primal envelope
    w1 = w1 + rho * (y1_new - v1_half);    % blur dual envelope
    w2 = w2 + rho * (y2_new - v2_half);    % x-gradient dual envelope
    w3 = w3 + rho * (y3_new - v3_half);    % y-gradient dual envelope

    % =================================================================
    % Convergence diagnostics (measured on the actual iterate, not z)
    % =================================================================

    % The actual primal iterate is always proj_[0,1](z), not z itself
    x_cur = boxProx(z);     % (H x W) current image estimate

    % Optional live display callback (used by the app for per-iteration display)
    if isfield(config, 'display_callback')
        config.display_callback(x_cur, k);
    end

    % Relative change: how much did the iterate move this step?
    % norm(v(:)) flattens v to a column vector before taking the 2-norm
    rel_change = norm(x_cur(:) - x_prev(:)) / (norm(x_prev(:)) + 1e-12);
    rel_history(k) = rel_change;   % store for output

    % Compute objective value: data fidelity + gamma * TV
    Kx = applyBlur(x_cur, k_hat);              % blurred estimate
    [gx, gy] = imageGrad(x_cur);               % discrete gradient of estimate
    tv_val = sum(sum(sqrt(gx.^2 + gy.^2)));    % isotropic TV: sum of pixel-wise norms

    if strcmp(problem, 'l1')
        fidelity = sum(abs(Kx(:) - b(:)));     % L1: sum of absolute residuals
    else
        fidelity = sum((Kx(:) - b(:)).^2);     % L2: sum of squared residuals
    end
    obj_history(k) = fidelity + gamma * tv_val;  % total objective

    % Print a status line every 50 iterations (and on the first one)
    if verbose && (k == 1 || mod(k, 50) == 0)
        fprintf('  iter %4d  obj=%.6e  rel_change=%.6e\n', ...
                k, obj_history(k), rel_change);
    end

    % --- Criterion 1: iterate-change convergence ---
    % Stop if the solution barely moved (and we've done at least 2 iterations)
    iterate_converged = (rel_change < tol) && (k > 1);

    % --- Criterion 2: objective plateau ---
    % For L1 (non-smooth), the iterate can oscillate slightly above tol even
    % when very close to the solution. We also stop if the objective hasn't
    % improved meaningfully over the last 20 iterations.
    OBJ_WINDOW = 20;
    obj_plateau = false;
    if k >= OBJ_WINDOW
        obj_win   = obj_history(k - OBJ_WINDOW + 1 : k);  % sliding window of values
        obj_range = max(obj_win) - min(obj_win);            % peak-to-peak variation
        % Plateau if relative variation is below tol
        obj_plateau = (obj_range / (abs(obj_history(k)) + 1e-12)) < tol;
    end

    % Stop if either criterion is met
    if iterate_converged || obj_plateau
        converged = true;
        if verbose
            fprintf('  Converged at iteration %d  (rel_change=%.2e)\n', k, rel_change);
        end
        break   % jump out of the for loop early
    end

    x_prev = x_cur;     % save current iterate for next iteration's comparison
end  % end for k

% -------------------------------------------------------------------------
% 5. Package outputs
% -------------------------------------------------------------------------

% Final solution: project the envelope z onto [0,1]
x_sol = boxProx(z);

elapsed = toc(t_start);     % stop timer, get elapsed seconds

if verbose
    fprintf('  Done -- %d iters, %.2fs, converged=%d\n', k, elapsed, converged);
end

% Pack everything into a struct (MATLAB's equivalent of a Python dict)
info.objective  = obj_history(1:k);     % trim unused preallocated zeros
info.rel_change = rel_history(1:k);
info.iterations = k;
info.converged  = converged;
info.time_sec   = elapsed;

end  % <<<< end of PDDR (main function)


% =========================================================================
%  LOCAL HELPER FUNCTIONS
%  These live in the same file and are only callable from within it.
%  MATLAB looks for local functions after the main function's closing `end`.
% =========================================================================

function k_hat = psfToOtf(psf, H, W)
% Convert a spatial PSF into the Fourier-domain OTF (eigenvalues of K).
% The PSF center must land at pixel (1,1) for the FFT convention to be correct.
[kH, kW] = size(psf);           % size of the kernel
padded = zeros(H, W);           % zero-pad to full image size
padded(1:kH, 1:kW) = psf;      % place PSF in top-left corner
% Shift so the kernel center moves to (1,1) -- this is the standard OTF trick.
% circshift(A, n, dim) shifts dimension dim by n positions (negative = shift left/up).
padded = circshift(padded, -floor(kH/2), 1);    % shift rows upward
padded = circshift(padded, -floor(kW/2), 2);    % shift cols leftward
k_hat  = fft2(padded);          % DFT of centered PSF = eigenvalues of K
end


function out = applyBlur(x, k_hat)
% Apply the blur operator K:  Kx = IFFT(k_hat .* FFT(x))
% Multiplication in Fourier domain = convolution in spatial domain.
out = real(ifft2(k_hat .* fft2(x)));   % real() removes numerical imaginary noise
end


function out = applyBlurAdj(x, k_hat)
% Apply the adjoint blur K^T:  K^T x = IFFT(conj(k_hat) .* FFT(x))
% Taking the complex conjugate of the eigenvalues transposes the operator.
out = real(ifft2(conj(k_hat) .* fft2(x)));
end


function [gx, gy] = imageGrad(x)
% Discrete forward finite differences with periodic (circular) boundary.
% gx(i,j) = x(i, j+1) - x(i, j)   -- horizontal difference (right neighbor)
% gy(i,j) = x(i+1, j) - x(i, j)   -- vertical difference   (bottom neighbor)
% circshift(A, -1, 2) shifts every column left by 1 (wraps the last column to front).
gx = circshift(x, -1, 2) - x;  % forward diff in x-direction (columns)
gy = circshift(x, -1, 1) - x;  % forward diff in y-direction (rows)
end


function out = imageDivAdj(gx, gy)
% Adjoint of imageGrad: this is the negative divergence with periodic BC.
% It's the "backward" finite difference, which is the adjoint of the forward one.
% circshift(A, 1, 2) shifts every column right by 1 (wraps the first column to end).
out = (circshift(gx, 1, 2) - gx) ...   % backward diff in x (adjoint of D_x)
    + (circshift(gy, 1, 1) - gy);       % backward diff in y (adjoint of D_y)
end


function [Ax1, Ax2, Ax3] = applyA(x, k_hat)
% Apply the stacked operator A = [K; D_x; D_y] to image x.
% Output has three components, one per channel.
Ax1 = applyBlur(x, k_hat);     % channel 1: blurred image  (Kx)
[Ax2, Ax3] = imageGrad(x);     % channel 2: horiz grad, channel 3: vert grad
end


function out = applyAT(y1, y2, y3, k_hat)
% Apply the adjoint A^T = [K^T | div] to the three dual channels.
% A^T [y1; y2; y3] = K^T y1 + div(y2, y3)
out = applyBlurAdj(y1, k_hat) ...   % adjoint blur applied to blur channel
    + imageDivAdj(y2, y3);           % adjoint gradient (divergence) on TV channels
end


function D = laplacianSymbol(H, W)
% Eigenvalues of D^T D (discrete Laplacian with periodic BC) in Fourier domain.
% For a 1D periodic difference of length N, the eigenvalues are 2 - 2*cos(2*pi*k/N).
% In 2D, we sum the horizontal and vertical contributions.
wx = (2*pi/W) * (0:W-1);       % horizontal frequencies: row vector (1 x W)
wy = (2*pi/H) * (0:H-1)';      % vertical frequencies:   col vector (H x 1)
% MATLAB broadcasts: wx (1xW) + wy (Hx1) -> (H x W) matrix automatically
D  = (2 - 2*cos(wx)) + (2 - 2*cos(wy));
end


function val = fieldOrDefault(s, field, default)
% Read a field from struct s if it exists; otherwise return default.
% isfield(s, field) returns true/false.
% s.(field) uses dynamic field access -- equivalent to s.gamma when field='gamma'.
if isfield(s, field)
    val = s.(field);    % field exists: return its value
else
    val = default;      % field missing: use the default
end
end


