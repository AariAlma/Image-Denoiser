function [xTrue, b, psf] = buildCorruptedImage(imagePath, varargin)
p = inputParser;
addRequired(p,  'imagePath');
addParameter(p, 'BlurKind',    'gaussian');
addParameter(p, 'BlurSize',    9);
addParameter(p, 'BlurSigma',   2.0);
addParameter(p, 'BlurRadius',  5);           % new
addParameter(p, 'MotionLen',   9);
addParameter(p, 'MotionAngle', 0.0);
addParameter(p, 'NoiseType',   'gaussian');
addParameter(p, 'NoiseLevel',  0.01);
addParameter(p, 'Seed',        0);
parse(p, imagePath, varargin{:});
opt = p.Results;

raw = imread(opt.imagePath);
if size(raw, 3) == 3
    raw = rgb2gray(raw);
end
xTrue = double(raw) / 255.0;
[nRows, nCols] = size(xTrue);

switch lower(opt.BlurKind)
    case 'gaussian'
        psf = gaussianPSF(opt.BlurSize, opt.BlurSigma);
    case 'motion'
        psf = motionPSF(opt.MotionLen, opt.MotionAngle);
    case 'disk'                               % new
        psf = fspecial('disk', opt.BlurRadius);
    case 'average'                            % new
        psf = fspecial('average', opt.BlurSize);
    otherwise
        error("buildCorruptedImage: unknown BlurKind '%s'.", opt.BlurKind);
end

[kH, kW] = size(psf);
padded = zeros(nRows, nCols);
padded(1:kH, 1:kW) = psf;
padded  = circshift(padded, -floor(kH/2), 1);
padded  = circshift(padded, -floor(kW/2), 2);
eigK    = fft2(padded);
blurred = real(ifft2(eigK .* fft2(xTrue)));

b = addNoise(blurred, opt.NoiseType, opt.NoiseLevel, opt.Seed);
end