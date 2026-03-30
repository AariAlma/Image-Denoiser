function [xSol,info] = primalDouglasRachford(problem,xinitial,kernel,b,i)
%Solves image deblurring via Primal Doughlas Rachford splitting alg
%
%INPUTS
% problem: specify "l1" or "l2"
% xinitial: Initial guess for image
% kernel: convolution kernel for blurring
% i: algorithm parameters
%     i.maxiter       - max iterations
%     i.tprimaldr     - Step size parameter t
%     i.rhoprimaldr   - Relaxation parameter rho (0 < rho < 2)
%     i.gammal1       - Regularization for l1 problem
%     i.gammal2       - Regularization for l2 problem
%     i.tol           - Convergence tolerance (default 1e-4)
%
%OUTPUTS
% xSol: deblurred/denoised image
% info: structure containing convergence information

%Image dimensions
[nrows,ncols] = size(b);
n = nrows*ncols;

% Extract algorithm parameters with defaults
if ~isfield(i, 'maxiter')
    i.maxiter = 500;  % Default maximum iterations
end

if ~isfield(i, 'tprimaldr')
    i.tprimaldr = 1.0;  % Default step size
end

if ~isfield(i, 'rhoprimaldr')
    i.rhoprimaldr = 1.0;  % Default relaxation parameter (1.0 is typical)
end

if ~isfield(i, 'tol')
    i.tol = 1e-4;  % Default convergence tolerance
end

% Extract regularization parameter based on problem type
if strcmp(problem, 'l1')
    if ~isfield(i, 'gammal1')
        i.gammal1 = 0.01;  % Default regularization for l1
    end
    gamma = i.gammal1;
elseif strcmp(problem, 'l2')
    if ~isfield(i, 'gammal2')
        i.gammal2 = 0.01;  % Default regularization for l2
    end
    gamma = i.gammal2;
else
    error('Problem type must be either ''l1'' or ''l2''');
end

t = i.tprimaldr;
rho = i.rhoprimaldr;
tol = i.tol;

%Checks
fprintf('Problem type: %s\n', problem);
fprintf('Image size: %d x %d (%d pixels)\n', nrows, ncols, n);
fprintf('Step size (t): %.4f\n', t);
fprintf('Relaxation parameter (rho): %.4f\n', rho);
fprintf('Regularization (gamma): %.6f\n', gamma);
fprintf('Maximum iterations: %d\n', i.maxiter);
fprintf('Convergence tolerance: %.2e\n', tol);

% Compute eigenvalue arrays
eigArry_K = eigValsForPeriodicConvOp(kernel, nrows, ncols);
eigArry_D1 = eigValsForPeriodicConvOp([-1,1]', nrows, ncols);
eigArry_D2 = eigValsForPeriodicConvOp([-1,1], nrows, ncols); 

% Compute transposes
eigArry_KTrans = conj(eigArry_K);
eigArry_D1Trans = conj(eigArry_D1);
eigArry_D2Trans = conj(eigArry_D2);

% Create function handles
applyK = @(x) applyPeriodicConv2D(x, eigArry_K); %Kx
applyKTrans = @(x) applyPeriodicConv2D(x, eigArry_KTrans); % K^Tx
%D1^Tx, D2^Tx
applyD1Trans = @(x) applyPeriodicConv2D(x, eigArry_D1Trans);
applyD2Trans = @(x) applyPeriodicConv2D(x, eigArry_D2Trans);
%D1x, D2x
applyD1 = @(x) applyPeriodicConv2D(x, eigArry_D1); 
applyD2 = @(x) applyPeriodicConv2D(x, eigArry_D2);  

%Eigenvalues of (I + A^TA)
eigValsMat = ones(nrows, ncols) + ...
             eigArry_KTrans.*eigArry_K + ...
             eigArry_D1Trans.*eigArry_D1 + ...
             eigArry_D2Trans.*eigArry_D2;

%Start with initilization at 0 for now
% Initialize z_1 (same size as image)
z1 = zeros(nrows, ncols);

% Initialize z_2 (3 blocks, each same size as image)
z2 = zeros(nrows, ncols, 3);

% Preallocate variables for the main loop 
x_k = zeros(nrows, ncols);
y_k = zeros(nrows, ncols, 3);
u_k = zeros(nrows, ncols);
v_k = zeros(nrows, ncols, 3);

% Initialize for convergence checking
x_k_prev = rand(nrows, ncols);
converged = false;

tic; 
for k = 1:i.maxiter
    % 1. Compute x^k = prox_tf(z_1)
    %    Resolvent of \mathcal{A}
    x_k = boxProx(z1);

    %2. Compute y^k = prox_tg(z_2)
    if strcmp(problem, 'l1')
        % Proximal operator of t*||y1 - b||_1
        y_k(:,:,1) = l1ShiftProx(z2(:,:,1), b, t);
        
    elseif strcmp(problem, 'l2')
        % Proximal operator of t*||y1 - b||_2^2
        y_k(:,:,1) = l2shiftprox(z2(:,:,1), b, t);
    end
    
    % Proximal operator of t*gamma*||(y2, y3)||_iso (total variation)
    [y_k(:,:,2), y_k(:,:,3)] = isoProx(z2(:,:,2), z2(:,:,3), t * gamma);

    %3. Compute u_k
    %   Resolvent of \mathcal{B}
    %   We are solving the equation (I+A^TA)u_k = RHS
    %First do A^T*(2y_k -z2)
    temp_y = 2*y_k - z2;
    %Apply A^T to temp_y
    ATrans_times_temp_y = applyKTrans(temp_y(:,:,1)) + ...
                      applyD1Trans(temp_y(:,:,2)) + ...
                      applyD2Trans(temp_y(:,:,3));
    %Then compute the whole RHS
    rhs = 2 * x_k - z1 + ATrans_times_temp_y;

    %Solve the linear system
    u_k = real(ifft2(fft2(rhs) ./ eigValsMat));

    %4. Compute v^k= Au^k
    v_k(:,:,1) = applyK(u_k);    % K * u^k
    v_k(:,:,2) = applyD1(u_k);   % D1 * u^k
    v_k(:,:,3) = applyD2(u_k);   % D2 * u^k

    %5. Update z1,z2
    z1 = z1 + rho * (u_k - x_k);
    z2 = z2 + rho * (v_k - y_k);

    % Check for convergence
    rel_change = norm(x_k(:) - x_k_prev(:)) / (norm(x_k_prev(:)) + eps);
    
    if rel_change < tol && k > 1
        fprintf('Converged at iteration %d (rel_change = %.6e)\n', k, rel_change);
        converged = true;
        break;
    end
    
    % Store current iterate for next comparison
    x_k_prev = x_k;

    if mod(k, 50) == 0 || k == 1
       fprintf('%4d\t %.4f\t rel_change = %.6e\n', k, toc, rel_change);
    end
end

% Compute final solution
xSol = boxProx(z1);

% Store info
totalTime = toc;
info.converged = converged;
info.iterations = k;
info.cpuTime = totalTime;
info.avgTimePerIter = totalTime / k;
info.algorithm = 'Primal Douglas-Rachford Splitting';
info.problemType = problem;

fprintf('\nAlgorithm completed: %s\n', info.algorithm);
fprintf('Converged: %s\n', mat2str(converged));
fprintf('Total iterations: %d\n', k);
fprintf('Total time: %.4f seconds\n', totalTime);

end