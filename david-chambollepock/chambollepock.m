function x = chambollepock(problem, x_initial, kernel, b, i)
% CHAMBOLLEPOCK: Solves the image deblurring problem using Algorithm 4.
% Implements the primal-dual saddle point solver for:
% min_x  g_1(Kx) + \gamma ||Dx||_{iso} + \delta_C(x)
% where g_1 is either the L1 norm or L2 squared norm based on 'problem'.

    % Start the CPU Timer to track execution time
    tic;

    % Extract step sizes and iteration limits from the settings struct
    maxiter = i.maxiter;
    t = i.tcp;
    s = i.scp;

    % Determine the regularization weight based on the chosen problem
    if strcmp(problem, 'l1')
        gamma = i.gammal1;
    elseif strcmp(problem, 'l2')
        gamma = i.gammal2;
    else
        error('Invalid problem type. Must be "l1" or "l2".');
    end

    % Initialize primal variables
    x = x_initial;
    z = x_initial; % Extrapolated variable z^0 = x^0 per algorithm spec

    % Initialize dual variables (y1 for blur, y2 & y3 for vertical/horizontal gradients)
    [rows, cols] = size(x_initial);
    y1 = zeros(rows, cols);
    y2 = zeros(rows, cols);
    y3 = zeros(rows, cols);

    % Main Iteration Loop
    for k = 1:maxiter

        % ==========================================
        % 1. DUAL UPDATE (y-step)
        % ==========================================
        % Compute v = y^{k-1} + s * A(z^{k-1})
        [Az1, Az2, Az3] = A_op(z, kernel);
        v1 = y1 + s * Az1;
        v2 = y2 + s * Az2;
        v3 = y3 + s * Az3;

        % Apply Moreau's Identity to compute the proximal operator of the conjugate:
        % prox_{s g*}(v) = v - s * prox_{g/s}(v/s)

        % Compute prox_{g1 / s}(v1 / s) for the Data Fidelity Term
        if strcmp(problem, 'l1')
            % L1 penalty: g1(u) = ||u - b||_1
            % Proximal mapping is a shifted soft-thresholding
            p1 = b + l1Prox(v1/s - b, 1/s);

        elseif strcmp(problem, 'l2')
            % L2 penalty: g1(u) = ||u - b||_2^2
            % Proximal mapping derived by setting derivative to zero
            p1 = (v1 + 2*b) / (s + 2);
        end

        % Compute prox_{g2 / s}(v2/s, v3/s) for the Isotropic TV Term
        % g2(u2, u3) = \gamma ||(u2, u3)||_{iso}
        [p2, p3] = isoProx(v2/s, v3/s, gamma/s);

        % Finalize the dual update using the results from Moreau's identity
        y1 = v1 - s * p1;
        y2 = v2 - s * p2;
        y3 = v3 - s * p3;

        % ==========================================
        % 2. PRIMAL UPDATE (x-step)
        % ==========================================
        % Compute x^k = prox_{t f}(x^{k-1} - t * A^T y^k)
        x_prev = x;

        % Apply the adjoint operator A^T to the dual variables
        Aty = At_op(y1, y2, y3, kernel);

        % Since f(x) = \delta_C(x), the proximal operator is a simple box projection
        x = boxProx(x_prev - t * Aty);

        % ==========================================
        % 3. EXTRAPOLATION (z-step)
        % ==========================================
        % z^k = 2x^k - x^{k-1}
        z = 2 * x - x_prev;

    end

    % End Timer
    cpu_time = toc;

    % Compute final objective value for reporting
    [Kx, D1x, D2x] = A_op(x, kernel);
    if strcmp(problem, 'l1')
        data_term = sum(abs(Kx(:) - b(:)));
    else
        data_term = sum((Kx(:) - b(:)).^2);
    end
    reg_term = gamma * sum(sqrt(D1x(:).^2 + D2x(:).^2));
    final_obj = data_term + reg_term;

    % Print Summary (Required by project spec)
    fprintf('\n--- Chambolle-Pock Summary ---\n');
    fprintf('Problem        : %s\n', problem);
    fprintf('Iterations     : %d\n', maxiter);
    fprintf('Final Objective: %.4f\n', final_obj);
    fprintf('CPU Time       : %.2f seconds\n', cpu_time);
    fprintf('------------------------------\n');
end
