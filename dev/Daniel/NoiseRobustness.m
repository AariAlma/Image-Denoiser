%% Noise Robustness: l1 vs l2 under Salt & Pepper
sp_levels = 0.02:0.04:0.50;
psnr_l1 = zeros(size(sp_levels));
psnr_l2 = zeros(size(sp_levels));

params.maxiter        = 500;
params.tprimaldr      = 0.5;
params.rhoprimaldr    = 1.0;
params.tol            = 1e-4;
params.record_metrics = true;

for ni = 1:length(sp_levels)
    [I_original, b, kernel] = buildCorruptedImage(imagePath, ...
        'BlurKind', 'gaussian', 'BlurSize', 9, 'BlurSigma', 2.0, ...
        'NoiseType', 'saltpepper', 'NoiseLevel', sp_levels(ni));

    params.I_reference = I_original;
    xinitial = zeros(size(b));

    % l1 fidelity
    params.gammal1 = 0.01;
    [~, info1] = primalDouglasRachford('l1', xinitial, kernel, b, params);
    psnr_l1(ni) = info1.psnr(end);

    % l2 fidelity
    params.gammal2 = 0.01;
    [~, info2] = primalDouglasRachford('l2', xinitial, kernel, b, params);
    psnr_l2(ni) = info2.psnr(end);

    fprintf('SP=%.2f  l1: %.2f dB   l2: %.2f dB\n', ...
        sp_levels(ni), psnr_l1(ni), psnr_l2(ni));
end

figure;
plot(sp_levels, psnr_l1, 'b-o', sp_levels, psnr_l2, 'r-s', 'LineWidth', 1.5);
xlabel('Salt & Pepper Density');
ylabel('PSNR (dB)');
legend('\ell_1 fidelity', '\ell_2 fidelity');
title('Noise Robustness: \ell_1 vs \ell_2');
grid on;