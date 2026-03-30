function [xTrue, b, psf] = buildCorruptedImage(imagePath, varargin)
%BUILDCORRUPTEDIMAGE  Full image corruption pipeline: load -> blur -> noise.
%
%   [xTrue, b, psf] = buildCorruptedImage(imagePath, Name, Value, ...)
%
%   INPUTS (required)
%     imagePath  - path to image file (JPEG, TIFF, PNG, ...)
%
%   NAME-VALUE OPTIONS
%     'BlurKind'    - 'gaussian' (default) or 'motion'
%     'BlurSize'    - kernel side length for Gaussian blur  (default: 9)
%     'BlurSigma'   - std dev for Gaussian blur             (default: 2.0)
%     'MotionLen'   - pixel length for motion blur          (default: 9)
%     'MotionAngle' - angle in degrees for motion blur      (default: 0.0)
%     'NoiseType'   - 'gaussian' (default) or 'saltpepper'
%     'NoiseLevel'  - noise strength (see addNoise)         (default: 0.01)
%     'Seed'        - RNG seed for noise                    (default: 0)
%
%   OUTPUTS
%     xTrue - H x W double, values in [0,1] — clean grayscale image
%     b     - H x W double, values in [0,1] — blurred + noisy observation
%     psf   - kH x kW double, sums to 1    — spatial blur kernel
%
%   EXAMPLE
%     [xTrue, b, psf] = buildCorruptedImage('cameraman.jpg', ...
%         'BlurSigma', 3.0, 'NoiseLevel', 0.02);

% --- Parse inputs ----------------------------------------------------------
p = inputParser;
addRequired(p,  'imagePath');
addParameter(p, 'BlurKind',    'gaussian');
addParameter(p, 'BlurSize',    9);
addParameter(p, 'BlurSigma',   2.0);
addParameter(p, 'MotionLen',   9);
addParameter(p, 'MotionAngle', 0.0);
addParameter(p, 'NoiseType',   'gaussian');
addParameter(p, 'NoiseLevel',  0.01);
addParameter(p, 'Seed',        0);
parse(p, imagePath, varargin{:});
opt = p.Results;

% --- Load image ------------------------------------------------------------
raw = imread(opt.imagePath);
if size(raw, 3) == 3
    raw = rgb2gray(raw);
end
xTrue = double(raw) / 255.0;
[nRows, nCols] = size(xTrue);

% --- Build blur kernel -----------------------------------------------------
switch lower(opt.BlurKind)
    case 'gaussian'
        psf = gaussianPSF(opt.BlurSize, opt.BlurSigma);
    case 'motion'
        psf = motionPSF(opt.MotionLen, opt.MotionAngle);
    otherwise
        error("buildCorruptedImage: unknown BlurKind '%s'.", opt.BlurKind);
end

% --- Apply blur via circular convolution (FFT) -----------------------------
% Uses shared utilities from Matlab_code/ (added to path automatically)
matlabCodeDir = fullfile(fileparts(mfilename('fullpath')), '..', 'Matlab_code');
addpath(matlabCodeDir);
eigK   = eigValsForPeriodicConvOp(psf, nRows, nCols);
blurred = real(applyPeriodicConv2D(xTrue, eigK));

% --- Add noise -------------------------------------------------------------
b = addNoise(blurred, opt.NoiseType, opt.NoiseLevel, opt.Seed);
end
