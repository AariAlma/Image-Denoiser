function psf = buildKernel(kind, varargin)
%BUILDKERNEL  Blur kernel factory.
%
%   psf = buildKernel('gaussian', size, sigma)
%   psf = buildKernel('motion',   length, angle)
%
%   INPUTS
%     kind  - 'gaussian' or 'motion'
%     For 'gaussian': size (int), sigma (float)
%     For 'motion':   length (int), angle in degrees (float)
%
%   OUTPUT
%     psf   - normalized blur kernel (sums to 1)

switch lower(kind)
    case 'gaussian'
        if numel(varargin) < 2
            error('buildKernel: gaussian requires (size, sigma).');
        end
        psf = gaussianPSF(varargin{1}, varargin{2});

    case 'motion'
        if numel(varargin) < 2
            error('buildKernel: motion requires (length, angle).');
        end
        psf = motionPSF(varargin{1}, varargin{2});

    otherwise
        error("buildKernel: unknown kind '%s'. Use 'gaussian' or 'motion'.", kind);
end
end
