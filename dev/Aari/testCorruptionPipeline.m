% testCorruptionPipeline.m
%
% HOW TO RUN:
%   1. Open MATLAB
%   2. Navigate to this folder via the address bar at the top, or run:
%        cd('D:\school\math563\project\Aari')
%   3. In the Command Window type:
%        testCorruptionPipeline
%
% You should see two figures and a stats printout in the Command Window.

imagePath = '../testimages/testimages/cameraman.jpg';

% --- Gaussian blur + Gaussian noise ----------------------------------------
[xTrue, b, psf] = buildCorruptedImage(imagePath, ...
    'BlurKind',   'gaussian', ...
    'BlurSize',   9,          ...
    'BlurSigma',  2.0,        ...
    'NoiseType',  'gaussian', ...
    'NoiseLevel', 0.02,       ...
    'Seed',       0);

fprintf('--- Gaussian blur + Gaussian noise ---\n');
fprintf('xTrue : %dx%d  range=[%.4f, %.4f]\n', size(xTrue,1), size(xTrue,2), min(xTrue(:)), max(xTrue(:)));
fprintf('b     : %dx%d  range=[%.4f, %.4f]\n', size(b,1),     size(b,2),     min(b(:)),     max(b(:)));
fprintf('psf   : %dx%d  sum=%.8f\n\n',          size(psf,1),   size(psf,2),   sum(psf(:)));

figure('Name', 'Gaussian blur + Gaussian noise');
subplot(1,3,1); imshow(xTrue);        title('Original');
subplot(1,3,2); imshow(b);            title('Blurred + Noisy');
subplot(1,3,3); imagesc(psf); axis image off; colormap hot; colorbar;
                title('PSF kernel');
sgtitle('Gaussian blur + Gaussian noise');

% --- Motion blur + salt-and-pepper noise -----------------------------------
[xTrue2, b2, psf2] = buildCorruptedImage(imagePath, ...
    'BlurKind',    'motion',     ...
    'MotionLen',   15,           ...
    'MotionAngle', 45,           ...
    'NoiseType',   'saltpepper', ...
    'NoiseLevel',  0.05,         ...
    'Seed',        0);

fprintf('--- Motion blur + salt-and-pepper noise ---\n');
fprintf('b     : %dx%d  range=[%.4f, %.4f]\n', size(b2,1),   size(b2,2),   min(b2(:)),   max(b2(:)));
fprintf('psf   : %dx%d  sum=%.8f\n',            size(psf2,1), size(psf2,2), sum(psf2(:)));

figure('Name', 'Motion blur + salt-and-pepper noise');
subplot(1,3,1); imshow(xTrue2);       title('Original');
subplot(1,3,2); imshow(b2);           title('Blurred + Noisy');
subplot(1,3,3); imagesc(psf2); axis image off; colormap hot; colorbar;
                title('PSF kernel');
sgtitle('Motion blur + salt-and-pepper noise');
