function x_rec = runAndVisualize(filename, problem, algorithm, kernel, noiseType, noiseParam, i)
% RUNANDVISUALIZE: Automates the process of loading, corrupting, solving,
% and graphing the results for the MATH 563 project.
%
% INPUTS:
%   filename   : String of the image filename (e.g., 'cameraman.jpg')
%   problem    : 'l1' or 'l2'
%   algorithm  : Name of the solver (e.g., 'chambollepock')
%   kernel     : The blur kernel matrix
%   noiseType  : 'salt & pepper' or 'gaussian'
%   noiseParam : Density (for S&P) or Variance (for Gaussian)
%   i          : The parameter struct for the solver

    fprintf('Loading and processing %s...\n', filename);

    % 1. Load and prepare the True Image
    I_true = imread(filename);
    if size(I_true, 3) == 3
        I_true = rgb2gray(I_true); % Convert to grayscale if it's RGB
    end
    I_true = double(I_true);

    % Normalize to [0, 1]
    mn = min(I_true(:));
    mx = max(I_true(:));
    I_true = (I_true - mn) / (mx - mn);

    % 2. Corrupt the Image (Blur + Noise)
    b_blurred = imfilter(I_true, kernel, 'circular', 'conv');

    % Add the requested noise
    if strcmp(noiseType, 'salt & pepper')
        b = imnoise(b_blurred, 'salt & pepper', noiseParam);
        noiseStr = sprintf('S&P (d=%.2f)', noiseParam);
    elseif strcmp(noiseType, 'gaussian')
        % Default mean is 0, noiseParam represents variance
        b = imnoise(b_blurred, 'gaussian', 0, noiseParam);
        noiseStr = sprintf('Gauss (var=%.3f)', noiseParam);
    else
        b = b_blurred;
        noiseStr = 'None';
    end

    % 3. Run the Solver (Solver only sees b, NEVER I_true)
    fprintf('Starting solver: %s (%s)...\n', algorithm, problem);
    x_rec = optsolver(problem, algorithm, b, kernel, b, i);

    % 4. Generate the Automated Figure
    % Extract the specific gamma used for the title
    if strcmp(problem, 'l1')
        gamma_val = i.gammal1;
    else
        gamma_val = i.gammal2;
    end

    % Create a wide figure
    figName = sprintf('Result: %s - %s', algorithm, filename);
    figure('Name', figName, 'Position', [100, 100, 1200, 450]);

    % Subplot 1: Ground Truth
    subplot(1, 3, 1);
    imshow(I_true, []);
    title('Ground Truth (Unseen)');

    % Subplot 2: Corrupted Observation
    subplot(1, 3, 2);
    imshow(b, []);
    title(sprintf('Corrupted Observation (b)\nNoise: %s', noiseStr));

    % Subplot 3: Reconstructed Output
    subplot(1, 3, 3);
    imshow(x_rec, []);
    title(sprintf('Reconstructed (%s, %s)\nGamma: %.3f | Iters: %d', ...
                  upper(algorithm), upper(problem), gamma_val, i.maxiter));

    % Add a super-title with algorithm step sizes (if applicable)
    if isfield(i, 'tcp') && isfield(i, 'scp')
        sgtitle(sprintf('Image: %s | Algorithm: %s | t=%.2f, s=%.2f', ...
                filename, upper(algorithm), i.tcp, i.scp), 'Interpreter', 'none');
    end
end
