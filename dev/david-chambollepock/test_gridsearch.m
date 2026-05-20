% TEST_GRIDSEARCH: Executes a combinatorial heuristic grid search for Chambolle-Pock
clear; clc; close all;

fprintf('Initializing Grid Search (729 combinations)...\n');

% ---------------------------------------------------------
% 1. Load and prepare the true image
% ---------------------------------------------------------
I_true = imread('cameraman.jpg');
if size(I_true, 3) == 3
    I_true = rgb2gray(I_true);
end
I_true = double(I_true);

% Normalize to [0, 1]
mn = min(I_true(:));
mx = max(I_true(:));
I_true = (I_true - mn) / (mx - mn);

% ---------------------------------------------------------
% 2. Define the Parameter Grid
% ---------------------------------------------------------
k_sizes  = [5, 15, 45];           % Small, Medium, Large
k_sigmas = [1.0, 5.0, 15.0];      % Low, Medium, High blur spread
noises   = [0.01, 0.05, 0.15];    % Low (1%), Medium (5%), High (15%) S&P Noise
gammas   = [0.005, 0.05, 0.5];    % L1 Regularization weight (orders of magnitude)
s_vals   = [0.05, 0.3, 0.9];      % Dual step sizes
t_vals   = [0.05, 0.3, 0.9];      % Primal step sizes

% Fix the number of iterations for the sweep to keep time reasonable
maxiter = 150;
problem_type = 'l2'; % We use L1 because we are testing Salt & Pepper noise

% ---------------------------------------------------------
% 3. Pre-allocate arrays for the final table (for speed)
% ---------------------------------------------------------
num_runs = length(k_sizes) * length(k_sigmas) * length(noises) * ...
           length(gammas) * length(s_vals) * length(t_vals);

col_ksize  = zeros(num_runs, 1);
col_sigma  = zeros(num_runs, 1);
col_noise  = zeros(num_runs, 1);
col_gamma  = zeros(num_runs, 1);
col_s      = zeros(num_runs, 1);
col_t      = zeros(num_runs, 1);
col_obj    = zeros(num_runs, 1);
col_mse    = zeros(num_runs, 1);
col_psnr   = zeros(num_runs, 1);

% ---------------------------------------------------------
% 4. Execute the Nested Grid Search
% ---------------------------------------------------------
idx = 1;
tic;

% Loop 1 & 2: Kernel Size and Sigma
for ks = k_sizes
    for sig = k_sigmas

        % Generate kernel and apply blur ONLY when kernel parameters change
        kernel = fspecial('gaussian', [ks, ks], sig);
        b_blurred = imfilter(I_true, kernel, 'circular', 'conv');

        % Loop 3: Noise Density
        for nz = noises

            % Apply noise ONLY when noise density changes
            b = imnoise(b_blurred, 'salt & pepper', nz);

            % Loop 4: Regularization Weight
            for g = gammas

                % Setup the parameters struct for optsolver
                i_struct.maxiter = maxiter;
                i_struct.gammal1 = g;

                % Loop 5 & 6: Step Sizes
                for s = s_vals
                    for t = t_vals

                        % Log the parameters for this specific run
                        col_ksize(idx) = ks;
                        col_sigma(idx) = sig;
                        col_noise(idx) = nz;
                        col_gamma(idx) = g;
                        col_s(idx)     = s;
                        col_t(idx)     = t;

                        % STABILITY CHECK: ||A||_2^2 is bounded by 9.
                        % We strictly need s * t * 9 < 1 for convergence.
                        if (s * t * 9) >= 1
                            % If unstable, skip solver to save time, log as NaN
                            col_obj(idx)  = NaN;
                            col_mse(idx)  = NaN;
                            col_psnr(idx) = NaN;
                        else
                            % Assign step sizes
                            i_struct.scp = s;
                            i_struct.tcp = t;

                            % Run the solver (capturing the objective history)
                            [x_rec, obj_history] = optsolver(problem_type, 'chambollepock', b, kernel, b, i_struct);

                            % Log the final objective value
                            col_obj(idx) = obj_history(end);

                            % Calculate Metrics against the UNSEEN true image
                            mse_val = mean((x_rec(:) - I_true(:)).^2);
                            col_mse(idx) = mse_val;

                            % PSNR Calculation (Max pixel value is 1.0)
                            if mse_val == 0
                                col_psnr(idx) = Inf;
                            else
                                col_psnr(idx) = 10 * log10(1^2 / mse_val);
                            end
                        end

                        % Print progress every 50 runs to avoid console bottleneck
                        if mod(idx, 50) == 0
                            fprintf('Completed %d / %d runs...\n', idx, num_runs);
                        end

                        idx = idx + 1;
                    end
                end
            end
        end
    end
end

total_time = toc;
fprintf('Grid Search Complete! Took %.2f seconds.\n', total_time);

##% ---------------------------------------------------------
##% 5. Build and Save the Table
##% ---------------------------------------------------------
##% Convert arrays into a structured MATLAB table
##ResultsTable = table(col_ksize, col_sigma, col_noise, col_gamma, col_s, col_t, col_obj, col_mse, col_psnr, ...
##    'VariableNames', {'KernelSize', 'Sigma', 'SP_Noise', 'Gamma', 'Step_s', 'Step_t', 'Final_Objective', 'MSE', 'PSNR'});
##
##% Save the table to a CSV file
##csv_filename = 'ChambollePock_GridSearch_Results.csv';
##writetable(ResultsTable, csv_filename);
##
##fprintf('Results successfully saved to: %s\n', csv_filename);
##
##% Display the top 5 best runs (Highest PSNR)
##fprintf('\n--- TOP 5 PARAMETER COMBINATIONS (Highest PSNR) ---\n');
##SortedTable = sortrows(ResultsTable, 'PSNR', 'descend');
##disp(head(SortedTable, 5));

% ---------------------------------------------------------
% 5. Build and Save the Table (Octave-compatible)
% ---------------------------------------------------------
% Convert arrays into a structured MATLAB table
ResultsTable = table(col_ksize, col_sigma, col_noise, col_gamma, col_s, col_t, col_obj, col_mse, col_psnr, ...
    'VariableNames', {'KernelSize', 'Sigma', 'SP_Noise', 'Gamma', 'Step_s', 'Step_t', 'Final_Objective', 'MSE', 'PSNR'});

csv_filename = 'ChambollePock_GridSearch_Results.csv';

% Save the table to a CSV file (with Octave fallback for writetable)
if exist('writetable', 'file')
    writetable(ResultsTable, csv_filename);
else
    % Octave fallback: Write headers then append numeric data manually
    fid = fopen(csv_filename, 'w');
    fprintf(fid, 'KernelSize,Sigma,SP_Noise,Gamma,Step_s,Step_t,Final_Objective,MSE,PSNR\n');

    % Concatenate columns and transpose for fprintf
    out_matrix = [col_ksize, col_sigma, col_noise, col_gamma, col_s, col_t, col_obj, col_mse, col_psnr]';
    fprintf(fid, '%g,%g,%g,%g,%g,%g,%g,%g,%g\n', out_matrix);
    fclose(fid);
end

fprintf('Results successfully saved to: %s\n', csv_filename);

% Display the top 5 best runs (Highest PSNR)
fprintf('\n--- TOP 5 PARAMETER COMBINATIONS (Highest PSNR) ---\n');
SortedTable = sortrows(ResultsTable, 'PSNR', 'descend');
disp(head(SortedTable, 5));
