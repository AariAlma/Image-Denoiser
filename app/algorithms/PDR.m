function [x_sol, info] = PDR(b, psf, config)
% PDR  Primal Douglas-Rachford solver for image deblurring.
%      Based on Daniel's implementation.
%
% Solves one of two problems depending on config.problem:
%
%   (L1)  min_x  ||Kx - b||_1   + delta_[0,1](x) + gamma * ||Dx||_iso
%   (L2)  min_x  ||Kx - b||_2^2 + delta_[0,1](x) + gamma * ||Dx||_iso
%
% KEY DIFFERENCE vs PDDR:
%   PDDR uses Primal-DUAL DR: it introduces separate dual variables and
%   applies Moreau's identity to evaluate prox_{t*g*} (prox of conjugate).
%
%   PDR uses Primal DR: both f and g are handled in the primal sense.
%   prox_{t*g} is evaluated directly (no Moreau flip), and the linear
%   system is (I + A^T A)^{-1}  rather than  (I + t^2 A^T A)^{-1}.
%   The step size t enters through the prox of g, not the linear solve.
%
% The algorithm maintains two envelope variables:
%   z1  -- primal envelope (image space)
%   z2  -- augmented envelope (range of A, 3 channels: blur, grad_x, grad_y)
%
% Each iteration:
%   x_k  = prox_{t*f}(z1)          -- box projection (primal)
%   y_k  = prox_{t*g}(z2)          -- data fidelity + TV prox (dual channels)
%   u_k  = (I + A^T A)^{-1} [2*x_k - z1 + A^T (2*y_k - z2)]  -- linear solve
%   v_k  = A * u_k                 -- forward operator applied to u_k
%   z1   = z1 + rho * (u_k - x_k) -- envelope update (primal)
%   z2   = z2 + rho * (v_k - y_k) -- envelope update (augmented)
%
% USAGE:
%   config.problem = 'l2';    % or 'l1'
%   config.gamma   = 0.006;
%   config.t       = 1.0;
%   config.rho     = 1.0;
%   config.maxiter = 500;
%   config.tol     = 1e-4;
%   config.verbose = true;
%   [x_sol, info] = PDR(b, psf, config);
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
t       = fieldOrDefault(config, 't',       1.0);    % step size (enters via prox of g)
rho     = fieldOrDefault(config, 'rho',     1.0);    % DR relaxation (0 < rho < 2)
maxiter = fieldOrDefault(config, 'maxiter', 500);    % max iterations
tol     = fieldOrDefault(config, 'tol',     1e-4);   % convergence tolerance
verbose = fieldOrDefault(config, 'verbose', true);   % print progress?

% -------------------------------------------------------------------------
% 2. Pre-compute fixed quantities
% -------------------------------------------------------------------------
[H, W] = size(b);           % image dimensions

% Compute OTF (Fourier-domain eigenvalues of blur operator K)
k_hat = psfToOtf(psf, H, W);   % (H x W) complex

% Denominator for the linear solve  (I + A^T A)^{-1}  in Fourier domain.
% Note: NO t^2 factor here -- that is the defining difference from PDDR.
%   Eigenvalues of A^T A = |k_hat|^2 (from K^T K) + laplacian (from D^T D)
denom = 1.0 + abs(k_hat).^2 + laplacianSymbol(H, W);   % (H x W) real

% -------------------------------------------------------------------------
% 3. Initialize variables
% -------------------------------------------------------------------------
[nrows, ncols] = deal(H, W);    % aliases matching Daniel's variable names

% z1: primal envelope in image space (H x W)
z1 = zeros(nrows, ncols);
if isfield(config, 'x_init') && ~isempty(config.x_init)
    z1 = config.x_init;    % use caller-supplied initial guess
end

% z2: augmented envelope in range of A (H x W x 3)
% z2(:,:,1) = blur channel, z2(:,:,2) = horiz grad, z2(:,:,3) = vert grad
% Using a 3D array (3rd dim = channel) is a MATLAB idiom for stacking matrices.
z2 = zeros(nrows, ncols, 3);

% Working copies for iterates
x_k      = zeros(nrows, ncols);    % current primal iterate
y_k      = zeros(nrows, ncols, 3); % current augmented iterate
u_k      = zeros(nrows, ncols);    % resolvent output (primal)
v_k      = zeros(nrows, ncols, 3); % resolvent output (augmented)
x_k_prev = b;                      % previous iterate (for convergence check)

obj_history = zeros(maxiter, 1);   % objective per iteration
rel_history = zeros(maxiter, 1);   % relative change per iteration

converged = false;

t_start = tic;  % start timer

% -------------------------------------------------------------------------
% 4. Main iteration loop
% -------------------------------------------------------------------------
for k = 1:maxiter

    % =====================================================================
    % STEP 1: x_k = prox_{t*f}(z1)
    %   f = delta_[0,1], so its prox is just clipping to [0,1].
    %   Note: the step size t does NOT appear here because the prox of an
    %   indicator function is always the projection, regardless of t.
    % =====================================================================
    x_k = boxProx(z1);      % project each pixel of z1 onto [0,1]

    % Optional live display callback (used by the app for per-iteration display)
    if isfield(config, 'display_callback')
        config.display_callback(x_k, k);
    end

    % =====================================================================
    % STEP 2: y_k = prox_{t*g}(z2)  -- applied channel by channel
    %
    %   Channel 1: g_1(y) = ||y - b||  (data fidelity, direct prox with step t)
    %   Channel 2&3: g_2(y2,y3) = gamma * iso(y2,y3)  (TV, direct prox)
    %
    %   CONTRAST WITH PDDR: here we compute prox_{t*g} directly.
    %   In PDDR, the dual uses Moreau's identity to get prox_{t*g*}.
    % =====================================================================
    if strcmp(problem, 'l1')
        % prox_{t * ||. - b||_1}(z2(:,:,1))  = b + softThresh(z2 - b, t)
        y_k(:,:,1) = l1ShiftProx(z2(:,:,1), b, t);
    else
        % prox_{t * ||. - b||_2^2}(z2(:,:,1))  = (z2 + 2*t*b) / (1 + 2*t)
        y_k(:,:,1) = l2shiftprox(z2(:,:,1), b, t);
    end
    % prox_{t*gamma * iso}(z2(:,:,2), z2(:,:,3))  -- isotropic shrinkage
    [y_k(:,:,2), y_k(:,:,3)] = isoProx(z2(:,:,2), z2(:,:,3), t * gamma);

    % =====================================================================
    % STEP 3: u_k = (I + A^T A)^{-1} [2*x_k - z1 + A^T(2*y_k - z2)]
    %   This is the resolvent of the "coupling" operator N = (A^T, -A).
    %   The reflected variables (2*x - z, 2*y - z2) are the DR reflections.
    %   The linear solve is diagonal in Fourier domain -> fast!
    % =====================================================================
    temp_y = 2 * y_k - z2;     % DR reflection of augmented variable (H x W x 3)

    % Compute A^T (2*y_k - z2): adjoint of A applied to all 3 channels
    ATtemp = applyAT(temp_y(:,:,1), temp_y(:,:,2), temp_y(:,:,3), k_hat);

    % Right-hand side of the linear system
    rhs = 2 * x_k - z1 + ATtemp;       % (H x W)

    % Solve via FFT: IFFT( FFT(rhs) ./ denom )
    u_k = real(ifft2(fft2(rhs) ./ denom));     % (H x W)

    % =====================================================================
    % STEP 4: v_k = A * u_k  (apply forward operator to get augmented output)
    % =====================================================================
    v_k(:,:,1) = applyBlur(u_k, k_hat);        % blur channel
    [v_k(:,:,2), v_k(:,:,3)] = imageGrad(u_k); % gradient channels

    % =====================================================================
    % STEP 5: Relaxed envelope update
    %   z1 moves toward u_k  (how far depends on rho)
    %   z2 moves toward v_k
    % =====================================================================
    z1 = z1 + rho * (u_k - x_k);       % primal envelope update
    z2 = z2 + rho * (v_k - y_k);       % augmented envelope update

    % =====================================================================
    % Convergence diagnostics
    % =====================================================================

    % Relative change in the primal iterate
    rel_change = norm(x_k(:) - x_k_prev(:)) / (norm(x_k_prev(:)) + eps);
    rel_history(k) = rel_change;

    % Compute objective at current iterate x_k
    Kx    = applyBlur(x_k, k_hat);                     % blurred estimate
    [gx, gy] = imageGrad(x_k);                          % gradient of estimate
    tv_val = sum(sum(sqrt(gx.^2 + gy.^2)));             % isotropic TV

    if strcmp(problem, 'l1')
        fidelity = sum(abs(Kx(:) - b(:)));              % L1 data fidelity
    else
        fidelity = sum((Kx(:) - b(:)).^2);              % L2 data fidelity
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

    x_k_prev = x_k;    % store current iterate for next iteration's comparison
end  % end for k

% -------------------------------------------------------------------------
% 5. Package outputs
% -------------------------------------------------------------------------
x_sol = boxProx(z1);    % final solution: project primal envelope onto [0,1]

elapsed = toc(t_start);

if verbose
    fprintf('  Done -- %d iters, %.2fs, converged=%d\n', k, elapsed, converged);
end

info.objective  = obj_history(1:k);
info.rel_change = rel_history(1:k);
info.iterations = k;
info.converged  = converged;
info.time_sec   = elapsed;

end  % <<<< end of PDR (main function)


% =========================================================================
%  LOCAL HELPER FUNCTIONS  (only callable from within this file)
% =========================================================================

function k_hat = psfToOtf(psf, H, W)
% Pad PSF to image size and shift center to (1,1), then FFT.
% This gives the Fourier-domain eigenvalues of the circular blur operator K.
[kH, kW] = size(psf);
padded = zeros(H, W);           % image-sized zero array
padded(1:kH, 1:kW) = psf;      % PSF in the top-left corner
padded = circshift(padded, -floor(kH/2), 1);    % shift center row to row 1
padded = circshift(padded, -floor(kW/2), 2);    % shift center col to col 1
k_hat  = fft2(padded);          % FFT -> eigenvalues of K
end


function out = applyBlur(x, k_hat)
% Kx = IFFT(k_hat .* FFT(x))  -- blur via element-wise multiply in Fourier domain
out = real(ifft2(k_hat .* fft2(x)));
end


function out = applyBlurAdj(x, k_hat)
% K^T x = IFFT(conj(k_hat) .* FFT(x))  -- adjoint (transpose) of blur
% Complex conjugate of eigenvalues = transpose of the operator
out = real(ifft2(conj(k_hat) .* fft2(x)));
end


function [gx, gy] = imageGrad(x)
% Forward finite differences (periodic boundary via circshift).
% circshift(x, -1, 2): shift left by 1 column (wrap last col to front)
gx = circshift(x, -1, 2) - x;  % horizontal: x(i, j+1) - x(i, j)
gy = circshift(x, -1, 1) - x;  % vertical:   x(i+1, j) - x(i, j)
end


function out = imageDivAdj(gx, gy)
% Adjoint of imageGrad (backward finite differences, periodic).
% circshift(x, 1, 2): shift right by 1 column (wrap first col to end)
out = (circshift(gx, 1, 2) - gx) ...   % adjoint of D_x
    + (circshift(gy, 1, 1) - gy);       % adjoint of D_y
end


function out = applyAT(y1, y2, y3, k_hat)
% A^T [y1; y2; y3] = K^T y1 + div(y2, y3)
out = applyBlurAdj(y1, k_hat) + imageDivAdj(y2, y3);
end


function D = laplacianSymbol(H, W)
% Fourier-domain eigenvalues of the discrete Laplacian D^T D (periodic BC).
wx = (2*pi/W) * (0:W-1);       % horizontal spatial frequencies (1 x W)
wy = (2*pi/H) * (0:H-1)';      % vertical spatial frequencies   (H x 1)
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


