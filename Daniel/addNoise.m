function noisy = addNoise(img, noiseType, noiseLevel, seed)
rng(seed);
noisy = img;
switch lower(noiseType)
    case 'gaussian'
        noisy = img + noiseLevel * randn(size(img));
    case 'saltpepper'
        nCorrupt = round(noiseLevel * numel(img));
        idx      = randperm(numel(img), nCorrupt);
        half     = floor(nCorrupt / 2);
        noisy(idx(1:half))     = 0;
        noisy(idx(half+1:end)) = 1;
    case 'speckle'                            % new
        noisy = img + img .* (noiseLevel * randn(size(img)));
    case 'poisson'                            % new
        noisy = imnoise(img, 'poisson');
    otherwise
        error("addNoise: unknown noiseType '%s'.", noiseType);
end
noisy = max(0, min(1, noisy));
end