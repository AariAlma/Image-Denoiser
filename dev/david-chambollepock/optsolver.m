function [x, varargout] = optsolver(problem, algorithm, x_initial, kernel, b, i, varargin)
% OPTSOLVER: Main wrapper function for image deblurring and denoising.
%
% INPUTS:
%   problem   : 'l1' (L1 data fidelity) or 'l2' (L2 data fidelity)
%   algorithm : Name of the splitting algorithm to use
%   x_initial : Initial guess for the image
%   kernel    : The blur kernel
%   b         : The observed blurred and noisy image
%   i         : Struct containing algorithm parameters
%   varargin  : Optional arguments (e.g., plot_filename)
%
% OUTPUT:
%   x         : The reconstructed image
%   varargout : Optional output containing the objective value history

    % Check for optional plot filename
    plot_filename = '';
    if nargin >= 7
        plot_filename = varargin{1};
    end

    % Route to the correct algorithm sub-routine
    if strcmp(algorithm, 'chambollepock')
        [x, obj_vals] = chambollepock(problem, x_initial, kernel, b, i, plot_filename);

    elseif strcmp(algorithm, 'douglasrachfordprimal')
        error('Primal Douglas-Rachford not yet implemented.');

    elseif strcmp(algorithm, 'douglasrachfordprimaldual')
        error('Primal-Dual Douglas-Rachford not yet implemented.');

    elseif strcmp(algorithm, 'admm')
        error('ADMM not yet implemented.');

    else
        error('Unknown algorithm: %s', algorithm);
    end

    % Optionally return objective values if requested
    if nargout > 1
        varargout{1} = obj_vals;
    end
end
