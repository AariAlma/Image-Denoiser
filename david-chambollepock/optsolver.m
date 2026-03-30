function x = optsolver(problem, algorithm, x_initial, kernel, b, i)
% OPTSOLVER: Main wrapper function for image deblurring and denoising.
%
% INPUTS:
%   problem   : 'l1' (L1 data fidelity) or 'l2' (L2 data fidelity)
%   algorithm : Name of the splitting algorithm to use
%   x_initial : Initial guess for the image (e.g., the blurred image b)
%   kernel    : The blur kernel (point spread function)
%   b         : The observed blurred and noisy image
%   i         : Struct containing algorithm parameters (maxiter, tcp, scp, etc.)
%
% OUTPUT:
%   x         : The reconstructed (deblurred and denoised) image

    % Route to the correct algorithm sub-routine
    if strcmp(algorithm, 'chambollepock')
        x = chambollepock(problem, x_initial, kernel, b, i);

    elseif strcmp(algorithm, 'douglasrachfordprimal')
        error('Primal Douglas-Rachford not yet implemented.');

    elseif strcmp(algorithm, 'douglasrachfordprimaldual')
        error('Primal-Dual Douglas-Rachford not yet implemented.');

    elseif strcmp(algorithm, 'admm')
        error('ADMM not yet implemented.');

    else
        error('Unknown algorithm: %s', algorithm);
    end

end
