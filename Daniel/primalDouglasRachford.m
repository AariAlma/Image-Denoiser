function [xSol, info] = primalDouglasRachford(problem, xinitial, kernel, b, i)
% Solves image deblurring via Primal Douglas-Rachford splitting algorithm
%
% INPUTS
%   problem:   'l1' or 'l2'
%   xinitial:  Initial guess for image
%   kernel:    Convolution kernel for blurring
%   b:         Observed (blurred + noisy) image
%   i:         Algorithm parameters struct
%     i.maxiter        - max iterations
%     i.tprimaldr      - step size t
%     i.rhoprimaldr    - relaxation parameter rho (0 < rho < 2)
%     i.gammal1        - regularization for l1 problem
%     i.gammal2        - regularization for l2 problem
%     i.tol            - convergence tolerance (default 1e-4)
%     i.record_metrics - set true to record objective & PSNR (default false)
%     i.I_reference    - ground truth image for PSNR (required if record_metrics=true)
%
% OUTPUTS
%   xSol: deblurred/denoised image
%   info: structure containing convergence info and metrics
%     info.objective  - objective value per iteration
%     info.psnr       - PSNR vs ground truth per iteration (if I_reference given)
%     info.iterations - total iterations run
%     info.converged  - whether tolerance was reached
%     info.cpuTime    - total runtime in seconds

%% Image dimensions
[nrows, ncols] = size(b);
n = nrows * ncols;

%% Extract parameters with defaults
if ~isfield(i, 'maxiter'),        i.maxiter = 500;         end
if ~isfield(i, 'tprimaldr'),      i.tprimaldr = 1.0;       end
if ~isfield(i, 'rhoprimaldr'),    i.rhoprimaldr = 1.0;     end
if ~isfield(i, 'tol'),            i.tol = 1e-4;            end
if ~isfield(i, 'record_metrics'), i.record_metrics = false; end

if strcmp(problem, 'l1')
    if ~isfield(i, 'gammal1'), i.gammal1 = 0.01; end
    gamma = i.gammal1;
elseif strcmp(problem, 'l2')
    if ~isfield(i, 'gammal2'), i.gammal2 = 0.01; end
    gamma = i.gammal2;
else
    error('Problem type must be either ''l1'' or ''l2''');
end

t   = i.tprimaldr;
rho = i.rhoprimaldr;
tol = i.tol;

%% Print summary
fprintf('Problem type: %s\n', problem);
fprintf('Image size: %d x %d (%d pixels)\n', nrows, ncols, n);
fprintf('Step size (t): %.4f\n', t);
fprintf('Relaxation parameter (rho): %.4f\n', rho);
fprintf('Regularization (gamma): %.6f\n', gamma);
fprintf('Maximum iterations: %d\n', i.maxiter);
fprintf('Convergence tolerance: %.2e\n', tol);

%% Preallocate metric arrays
if i.record_metrics
    obj_vals  = zeros(i.maxiter, 1);
    psnr_vals = zeros(i.maxiter, 1);
    ssim_vals = zeros(i.maxiter, 1);
    have_ref  = isfield(i, 'I_reference');
    if ~have_ref
        warning('record_metrics=true but no I_reference provided. PSNR will be skipped.');
    end
end

%% Compute eigenvalue arrays
eigArry_K  = eigValsForPeriodicConvOp(kernel,  nrows, ncols);
eigArry_D1 = eigValsForPeriodicConvOp([-1;1],  nrows, ncols);
eigArry_D2 = eigValsForPeriodicConvOp([-1, 1], nrows, ncols);

eigArry_KTrans  = conj(eigArry_K);
eigArry_D1Trans = conj(eigArry_D1);
eigArry_D2Trans = conj(eigArry_D2);

%% Function handles
applyK       = @(x) applyPeriodicConv2D(x, eigArry_K);
applyKTrans  = @(x) applyPeriodicConv2D(x, eigArry_KTrans);
applyD1      = @(x) applyPeriodicConv2D(x, eigArry_D1);
applyD1Trans = @(x) applyPeriodicConv2D(x, eigArry_D1Trans);
applyD2      = @(x) applyPeriodicConv2D(x, eigArry_D2);
applyD2Trans = @(x) applyPeriodicConv2D(x, eigArry_D2Trans);

%% Eigenvalues of (I + A^T A)
eigValsMat = ones(nrows, ncols) + ...
             eigArry_KTrans .* eigArry_K  + ...
             eigArry_D1Trans .* eigArry_D1 + ...
             eigArry_D2Trans .* eigArry_D2;

%% Initialise variables
z1 = zeros(nrows, ncols);
z2 = zeros(nrows, ncols, 3);

x_k      = zeros(nrows, ncols);
y_k      = zeros(nrows, ncols, 3);
u_k      = zeros(nrows, ncols);
v_k      = zeros(nrows, ncols, 3);
x_k_prev = rand(nrows, ncols);

converged = false;

%% Main loop
tic;
for k = 1:i.maxiter

    %--- Step 1: x^k = prox_tf(z1)  [box constraint] ---
    x_k = boxProx(z1);

    %--- Step 2: y^k = prox_tg(z2) ---
    if strcmp(problem, 'l1')
        y_k(:,:,1) = l1ShiftProx(z2(:,:,1), b, t);
    elseif strcmp(problem, 'l2')
        y_k(:,:,1) = l2ShiftProx(z2(:,:,1), b, t);
    end
    [y_k(:,:,2), y_k(:,:,3)] = isoProx(z2(:,:,2), z2(:,:,3), t * gamma);

    %--- Step 3: u^k = resolvent of B ---
    temp_y = 2 * y_k - z2;
    ATrans_times_temp_y = applyKTrans(temp_y(:,:,1)) + ...
                          applyD1Trans(temp_y(:,:,2)) + ...
                          applyD2Trans(temp_y(:,:,3));
    rhs = 2 * x_k - z1 + ATrans_times_temp_y;
    u_k = real(ifft2(fft2(rhs) ./ eigValsMat));

    %--- Step 4: v^k = A u^k ---
    v_k(:,:,1) = applyK(u_k);
    v_k(:,:,2) = applyD1(u_k);
    v_k(:,:,3) = applyD2(u_k);

    %--- Step 5: update z1, z2 ---
    z1 = z1 + rho * (u_k - x_k);
    z2 = z2 + rho * (v_k - y_k);

    %--- Record metrics ---
    if i.record_metrics
        % Objective: ||Kx - b||_1 + gamma * TV(x)
        Kx            = applyK(x_k);
        d1x           = applyD1(x_k);
        d2x           = applyD2(x_k);
        data_fidelity = sum(abs(Kx(:) - b(:)));
        tv_term       = gamma * sum(sqrt(d1x(:).^2 + d2x(:).^2));
        obj_vals(k)   = data_fidelity + tv_term;

        % PSNR vs ground truth 
        if have_ref
            N            = nrows * ncols;
            mse          = norm(x_k(:) - i.I_reference(:))^2;
            psnr_vals(k) = 10 * log10(N / (mse + eps));
        end

        % SSIM vs ground truth
        if have_ref
            ssim_vals(k) = computeSSIM(i.I_reference, x_k);
end
    end

    %--- Convergence check ---
    rel_change = norm(x_k(:) - x_k_prev(:)) / (norm(x_k_prev(:)) + eps);
    if rel_change < tol && k > 1
        fprintf('Converged at iteration %d (rel_change = %.6e)\n', k, rel_change);
        converged = true;
        break;
    end
    x_k_prev = x_k;

    if mod(k, 50) == 0 || k == 1
        fprintf('%4d\t %.4f s\t rel_change = %.6e\n', k, toc, rel_change);
    end
end

%% Final solution
xSol = boxProx(z1);

%% Store info
totalTime           = toc;
info.converged      = converged;
info.iterations     = k;
info.cpuTime        = totalTime;
info.avgTimePerIter = totalTime / k;
info.algorithm      = 'Primal Douglas-Rachford Splitting';
info.problemType    = problem;

% Trim metric arrays to actual number of iterations run
if i.record_metrics
    info.objective = obj_vals(1:k);
    if have_ref
        info.psnr = psnr_vals(1:k);
        info.ssim = ssim_vals(1:k);
    else
        info.psnr = [];
        info.ssim = [];
    end
else
    info.objective = [];
    info.psnr      = [];
    info.ssim      = [];
end

fprintf('\nAlgorithm completed: %s\n', info.algorithm);
fprintf('Converged: %s\n', mat2str(converged));
fprintf('Total iterations: %d\n', k);
fprintf('Total time: %.4f seconds\n', totalTime);
end