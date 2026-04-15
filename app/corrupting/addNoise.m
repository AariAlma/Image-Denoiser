function noisy = addNoise(img, noiseType, noiseLevel, seed)
%ADDNOISE  Add noise to an image and clip the result to [0, 1].
%
%   noisy = addNoise(img, noiseType, noiseLevel, seed)
%
%   INPUTS
%     img        - double matrix, values in [0, 1]
%     noiseType  - 'gaussian'   : additive white Gaussian noise (sigma)
%                  'saltpepper' : random pixels set to 0 or 1 (fraction)
%                  'speckle'    : multiplicative Gaussian noise (sigma)
%                  'poisson'    : photon shot noise (peak = 1/noiseLevel)
%                  'uniform'    : additive uniform noise in [-level, +level]
%     noiseLevel - noise strength (interpretation depends on noiseType)
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

    case 'speckle'
        % Multiplicative Gaussian: noisy = img * (1 + sigma*randn)
        noisy = img .* (1 + noiseLevel * randn(size(img)));

    case 'poisson'
        % Photon shot noise (Gaussian approximation, no toolbox required).
        % For a pixel with value p observed at `peak` counts, the shot noise
        % standard deviation is sqrt(p*peak)/peak = sqrt(p/peak).
        % noiseLevel is 1/peak (higher level = fewer photons = more noise).
        peak  = max(1, round(1 / max(noiseLevel, 1e-6)));
        noisy = img + sqrt(max(img, 0) / peak) .* randn(size(img));

    case 'uniform'
        % Additive uniform noise in [-noiseLevel, +noiseLevel]
        noisy = img + noiseLevel * (2 * rand(size(img)) - 1);

    otherwise
        error("addNoise: unknown noiseType '%s'.", noiseType);
end

noisy = max(0, min(1, noisy));
end
