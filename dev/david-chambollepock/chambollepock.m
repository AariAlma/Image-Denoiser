function [x, obj_vals] = chambollepock(problem, x_initial, kernel, b, i, varargin)
% CHAMBOLLEPOCK: Solves the image deblurring problem using Algorithm 4.

    % Check for optional plot filename
    plot_filename = '';
    if nargin >= 6
        plot_filename = varargin{1};
    end

    tic;

    maxiter = i.maxiter;
    t = i.tcp;
    s = i.scp;

    if strcmp(problem, 'l1')
        gamma = i.gammal1;
    elseif strcmp(problem, 'l2')
        gamma = i.gammal2;
    else
        error('Invalid problem type. Must be "l1" or "l2".');
    end

    x = x_initial;
    z = x_initial;

    [rows, cols] = size(x_initial);
    y1 = zeros(rows, cols);
    y2 = zeros(rows, cols);
    y3 = zeros(rows, cols);

    % Pre-allocate array for objective values
    obj_vals = zeros(maxiter, 1);

    for k = 1:maxiter

        % --- DUAL UPDATE (y-step) ---
        [Az1, Az2, Az3] = A_op(z, kernel);
        v1 = y1 + s * Az1;
        v2 = y2 + s * Az2;
        v3 = y3 + s * Az3;

        if strcmp(problem, 'l1')
            p1 = b + l1Prox(v1/s - b, 1/s);
        elseif strcmp(problem, 'l2')
            % CORRECTED: L2 penalty g1(u) = 0.5 * ||u - b||_2^2
            % prox_{g1/s}(v1/s) = (v1 + b) / (s + 1)
            p1 = (v1 + b) / (s + 1);
        end

        [p2, p3] = isoProx(v2/s, v3/s, gamma/s);

        y1 = v1 - s * p1;
        y2 = v2 - s * p2;
        y3 = v3 - s * p3;

        % --- PRIMAL UPDATE (x-step) ---
        x_prev = x;
        Aty = At_op(y1, y2, y3, kernel);
        x = boxProx(x_prev - t * Aty);

        % --- EXTRAPOLATION (z-step) ---
        z = 2 * x - x_prev;

        % --- OBJECTIVE CALCULATION ---
        % Track objective value for convergence plotting
        [Kx, D1x, D2x] = A_op(x, kernel);
        if strcmp(problem, 'l1')
            data_term = sum(abs(Kx(:) - b(:)));
        else
            data_term = 0.5 * sum((Kx(:) - b(:)).^2); % Matches Eq 2 and 3
        end
        reg_term = gamma * sum(sqrt(D1x(:).^2 + D2x(:).^2));

        obj_vals(k) = data_term + reg_term;
    end

    cpu_time = toc;

    % Print Summary
    fprintf('\n--- Chambolle-Pock Summary ---\n');
    fprintf('Problem        : %s\n', problem);
    fprintf('Iterations     : %d\n', maxiter);
    fprintf('Final Objective: %.4f\n', obj_vals(end));
    fprintf('CPU Time       : %.2f seconds\n', cpu_time);
    fprintf('------------------------------\n');

    % --- PLOTTING LOGIC ---
    % If a filename was provided, generate and save the convergence plot
    if ~isempty(plot_filename)
        % Create an invisible figure so it doesn't interrupt workflow
        fig = figure('Visible', 'off', 'Position', [100, 100, 600, 600]);
        plot(1:maxiter, obj_vals, 'b-', 'LineWidth', 2);
        grid on;
        xlabel('Iteration (k)', 'FontWeight', 'bold');
        ylabel('Objective Value', 'FontWeight', 'bold');

        % Create a descriptive title with all parameters
        plot_title = sprintf('Chambolle-Pock Convergence (%s Data Fidelity)\n\\gamma = %.3f, t = %.2f, s = %.2f', ...
                              upper(problem), gamma, t, s);
        title(plot_title, 'FontSize', 14);

        % Save to disk and close the hidden figure
        saveas(fig, plot_filename);
        close(fig);
        fprintf('Saved convergence plot to: %s\n', plot_filename);
    end
end
