function noisy = addNoise(img, noiseType, noiseLevel, seed)
%ADDNOISE  Add noise to an image and clip the result to [0, 1].
%
%   noisy = addNoise(img, noiseType, noiseLevel, seed)
%
%   INPUTS
%     img        - double matrix, values in [0, 1]
%     noiseType  - 'gaussian'   : additive white Gaussian noise
%                  'saltpepper' : random pixels set to 0 or 1
%     noiseLevel - for 'gaussian':   standard deviation of the noise
%                  for 'saltpepper': fraction of pixels corrupted
%     seed       - integer RNG seed for reproducibility
%
%   OUTPUT
%     noisy      - same size as img, values clipped to [0, 1]

rng(seed);
noisy = img;

switch lower(noiseType)
    case 'gaussian'
        noisy = img + noiseLevel * randn(size(img));

    case 'saltpepper'
        nCorrupt = round(noiseLevel * numel(img));
        idx      = randperm(numel(img), nCorrupt);
        half     = floor(nCorrupt / 2);
        noisy(idx(1:half))       = 0;   % pepper
        noisy(idx(half+1:end))   = 1;   % salt

    otherwise
        error("addNoise: unknown noiseType '%s'. Use 'gaussian' or 'saltpepper'.", noiseType);
end

noisy = max(0, min(1, noisy));
end
