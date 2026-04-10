function DeblurApp()
% DeblurApp  MATLAB programmatic UI for MATH 563 image deblurring project.
%
% Three-tab interface:
%   Tab 1  "Corrupt Image"   -- load a clean image, apply blur + noise
%   Tab 2  "Deblur/Denoise"  -- run PDR / PDDR / ADMM / CP with live display
%   Tab 3  "Analysis"        -- GT vs reconstruction, convergence plot, PSNR
%
% USAGE:
%   DeblurApp()

% ---- Add all app sub-folders to MATLAB path --------------------------------
% Use which() as a fallback — more robust than mfilename when the function
% is called from a different working directory or via a path entry.
appDir = fileparts(mfilename('fullpath'));
if isempty(appDir)
    appDir = fileparts(which('DeblurApp'));
end
addpath(fullfile(appDir, 'algorithms'));
addpath(fullfile(appDir, 'prox_operators'));
addpath(fullfile(appDir, 'corrupting'));
addpath(fullfile(appDir, 'utils'));

% ---- Main figure -----------------------------------------------------------
fig = uifigure('Name',     'MATH 563 — Image Deblurring App', ...
               'Position', [50, 50, 1400, 750]);

% ---- Shared app state (stored in fig.UserData) -----------------------------
app.img_original  = [];      % H×W double, ground truth (clean image)
app.img_blurred   = [];      % H×W double, corrupted observation b
app.img_recon     = [];      % H×W double, latest solver output
app.psf           = [];      % kH×kW double, current blur kernel ([] = unknown)
app.info          = [];      % struct returned by last solver call
app.imagePath     = '';      % full path to last loaded file
app.solverRunning = false;   % true while solver is executing (prevents re-entry)
fig.UserData = app;

% ---- Build tabs ------------------------------------------------------------
tg = uitabgroup(fig, 'Position', [0, 0, 1400, 750]);
buildCorruptTab(fig, tg);
buildDeblurTab(fig, tg);
buildAnalysisTab(fig, tg);

end  % <<<< end of DeblurApp (entry point)


% =============================================================================
%  TAB 1 — CORRUPT IMAGE
%  Layout: [ax_clean | ax_corrupted | options panel]
% =============================================================================

function buildCorruptTab(fig, tg)

tab = uitab(tg, 'Title', 'Corrupt Image');

% Two display axes (side by side, each ~470 px wide)
ax_clean = uiaxes(tab, 'Position', [10, 10, 465, 695]);
title(ax_clean, 'Clean Image');
ax_clean.XTick = []; ax_clean.YTick = [];

ax_corrupted = uiaxes(tab, 'Position', [485, 10, 465, 695]);
title(ax_corrupted, 'Corrupted Image');
ax_corrupted.XTick = []; ax_corrupted.YTick = [];

% Right options panel (~420 px wide)
panel = uipanel(tab, 'Position', [960, 10, 428, 695]);

% Save axes into app state before building panel controls
app = fig.UserData;
app.ax_clean     = ax_clean;
app.ax_corrupted = ax_corrupted;
fig.UserData = app;

buildCorruptPanel(fig, panel);

end


function buildCorruptPanel(fig, panel)
% Stacks controls top-to-bottom inside Tab 1's right panel.

PW = 412;   % usable panel width (pixels)
y  = 655;   % y cursor, decrements as rows are added

% ===== Load Clean Image =====
uilabel(panel, 'Text', 'Load Clean Image', ...
        'Position', [5, y, PW, 20], 'FontWeight', 'bold');
y = y - 30;

uibutton(panel, 'Text', 'Browse...', ...
         'Position', [5, y, 90, 26], ...
         'ButtonPushedFcn', @(~,~) cb_BrowseClean(fig));
lbl_path = uilabel(panel, ...
    'Text', 'No file selected', ...
    'Position', [103, y, PW - 105, 26], ...
    'FontAngle', 'italic', 'WordWrap', 'on');
y = y - 46;

% ===== Blur Kernel =====
uilabel(panel, 'Text', 'Blur Kernel', ...
        'Position', [5, y, PW, 20], 'FontWeight', 'bold');
y = y - 30;

uilabel(panel, 'Text', 'Type:', 'Position', [5, y, 45, 24]);
dd_blur = uidropdown(panel, ...
                     'Items', {'Gaussian', 'Motion'}, ...
                     'Value', 'Gaussian', ...
                     'Position', [55, y, 140, 26], ...
                     'ValueChangedFcn', @(~,~) cb_BlurTypeChanged(fig));
y = y - 36;

% --- Gaussian-specific controls ---
y_blur_row1 = y;    % save y so Motion controls can share same rows
lbl_ksize = uilabel(panel, 'Text', 'Kernel size:', 'Position', [5, y, 85, 24]);
ef_ksize  = uieditfield(panel, 'numeric', ...
                        'Value', 9, 'Limits', [1, 99], ...
                        'RoundFractionalValues', 'on', ...
                        'Position', [95, y, 70, 24]);
y = y - 34;
y_blur_row2 = y;

lbl_sigma = uilabel(panel, 'Text', 'Sigma:', 'Position', [5, y, 85, 24]);
ef_sigma  = uieditfield(panel, 'numeric', ...
                        'Value', 2.0, 'Limits', [0.1, 100], ...
                        'Position', [95, y, 70, 24]);
y = y - 46;

% --- Motion-specific controls (overlay the same rows, hidden by default) ---
lbl_mlen  = uilabel(panel, 'Text', 'Length (px):', ...
                    'Position', [5, y_blur_row1, 85, 24], 'Visible', 'off');
ef_mlen   = uieditfield(panel, 'numeric', ...
                        'Value', 15, 'Limits', [1, 500], ...
                        'RoundFractionalValues', 'on', ...
                        'Position', [95, y_blur_row1, 70, 24], 'Visible', 'off');
lbl_mangle = uilabel(panel, 'Text', 'Angle (°):', ...
                     'Position', [5, y_blur_row2, 85, 24], 'Visible', 'off');
ef_mangle  = uieditfield(panel, 'numeric', ...
                         'Value', 45, 'Limits', [-360, 360], ...
                         'Position', [95, y_blur_row2, 70, 24], 'Visible', 'off');

% ===== Noise =====
uilabel(panel, 'Text', 'Noise', ...
        'Position', [5, y, PW, 20], 'FontWeight', 'bold');
y = y - 30;

uilabel(panel, 'Text', 'Type:', 'Position', [5, y, 45, 24]);
dd_noise = uidropdown(panel, ...
                      'Items', {'Gaussian', 'Salt & Pepper'}, ...
                      'Value', 'Gaussian', ...
                      'Position', [55, y, 155, 26]);
y = y - 36;

uilabel(panel, 'Text', 'Level:', 'Position', [5, y, 45, 24]);
ef_nlevel = uieditfield(panel, 'numeric', ...
                        'Value', 0.02, 'Limits', [0, 1], ...
                        'Position', [55, y, 90, 24]);
y = y - 52;

% ===== Apply Corruption =====
uibutton(panel, 'Text', 'Apply Corruption', ...
         'Position', [5, y, 190, 34], ...
         'BackgroundColor', [0.18, 0.50, 0.82], ...
         'FontColor', 'white', 'FontWeight', 'bold', ...
         'ButtonPushedFcn', @(~,~) cb_ApplyCorruption(fig));
y = y - 46;

% ===== Send to Deblur tab =====
uibutton(panel, 'Text', 'Use for Deblurring  >', ...
         'Position', [5, y, 190, 28], ...
         'ButtonPushedFcn', @(~,~) cb_UseCorrupted(fig));
y = y - 40;

% ===== Status =====
lbl_status_c = uilabel(panel, 'Text', 'Ready.', ...
                        'Position', [5, y, PW, 22], 'FontAngle', 'italic');

% ---- Store all control handles in app state --------------------------------
app = fig.UserData;
app.lbl_imagepath_c = lbl_path;
app.dd_blur_c       = dd_blur;
app.dd_noise_c      = dd_noise;
app.ef_ksize        = ef_ksize;
app.ef_sigma        = ef_sigma;
app.ef_mlen         = ef_mlen;
app.ef_mangle       = ef_mangle;
app.ef_nlevel       = ef_nlevel;
app.lbl_ksize       = lbl_ksize;
app.lbl_sigma       = lbl_sigma;
app.lbl_mlen        = lbl_mlen;
app.lbl_mangle      = lbl_mangle;
app.lbl_status_c    = lbl_status_c;
fig.UserData = app;

end


% =============================================================================
%  TAB 2 — DEBLUR / DENOISE
%  Layout: [ax_blurred | ax_recon | options panel]
% =============================================================================

function buildDeblurTab(fig, tg)

tab = uitab(tg, 'Title', 'Deblur / Denoise');

ax_blurred = uiaxes(tab, 'Position', [10, 10, 305, 695]);
title(ax_blurred, 'Corrupted Input');
ax_blurred.XTick = []; ax_blurred.YTick = [];

ax_recon = uiaxes(tab, 'Position', [325, 10, 305, 695]);
title(ax_recon, 'Reconstruction (live)');
ax_recon.XTick = []; ax_recon.YTick = [];

panel = uipanel(tab, 'Position', [640, 10, 750, 695]);

app = fig.UserData;
app.ax_blurred = ax_blurred;
app.ax_recon   = ax_recon;
fig.UserData   = app;

buildDeblurPanel(fig, panel);

end


function buildDeblurPanel(fig, panel)
% Controls inside Tab 2's options panel.

PW = 734;
y  = 655;

% ===== Load Blurred Image =====
uilabel(panel, 'Text', 'Load Blurred Image', ...
        'Position', [5, y, PW, 20], 'FontWeight', 'bold');
y = y - 30;

uibutton(panel, 'Text', 'Browse...', ...
         'Position', [5, y, 90, 26], ...
         'ButtonPushedFcn', @(~,~) cb_BrowseBlurred(fig));
lbl_imgpath = uilabel(panel, ...
    'Text', 'No file  (or use Corrupt tab → "Use for Deblurring")', ...
    'Position', [103, y, PW - 105, 26], 'FontAngle', 'italic');
y = y - 44;

% ===== Ground Truth (for PSNR) =====
uilabel(panel, 'Text', 'Ground Truth  (for PSNR)', ...
        'Position', [5, y, PW, 20], 'FontWeight', 'bold');
y = y - 30;

uibutton(panel, 'Text', 'Browse GT...', ...
         'Position', [5, y, 100, 26], ...
         'ButtonPushedFcn', @(~,~) cb_BrowseGroundTruth(fig));
lbl_gtpath = uilabel(panel, ...
    'Text', 'None  (auto-set from Corrupt tab)', ...
    'Position', [113, y, PW - 115, 26], 'FontAngle', 'italic');
y = y - 46;

% ===== Algorithm =====
uilabel(panel, 'Text', 'Algorithm', ...
        'Position', [5, y, PW, 20], 'FontWeight', 'bold');
y = y - 32;

uilabel(panel, 'Text', 'Solver:', 'Position', [5, y, 55, 24]);
dd_alg = uidropdown(panel, ...
                    'Items', {'PDDR', 'PDR', 'ADMM', 'CP'}, ...
                    'Value', 'PDDR', ...
                    'Position', [65, y, 110, 26], ...
                    'ValueChangedFcn', @(~,~) cb_AlgorithmChanged(fig));

uilabel(panel, 'Text', 'Problem:', 'Position', [200, y, 65, 24]);
dd_prob = uidropdown(panel, ...
                     'Items', {'L2', 'L1'}, ...
                     'Value', 'L2', ...
                     'Position', [270, y, 80, 26]);
y = y - 46;

% ===== Parameters =====
uilabel(panel, 'Text', 'Parameters', ...
        'Position', [5, y, PW, 20], 'FontWeight', 'bold');
y = y - 32;

% Row 1: gamma  |  t (primal step)
uilabel(panel, 'Text', 'gamma:', 'Position', [5, y, 55, 24]);
ef_gamma = uieditfield(panel, 'numeric', ...
                       'Value', 0.006, 'Limits', [0, Inf], ...
                       'Position', [65, y, 95, 24]);

uilabel(panel, 'Text', 't (primal):', 'Position', [180, y, 78, 24]);
ef_t = uieditfield(panel, 'numeric', ...
                   'Value', 1.0, 'Limits', [0, Inf], ...
                   'Position', [263, y, 95, 24]);
y = y - 36;

% Row 2: rho OR s (dual)  |  maxiter
lbl_rho = uilabel(panel, 'Text', 'rho:', 'Position', [5, y, 55, 24]);
ef_rho  = uieditfield(panel, 'numeric', ...
                      'Value', 1.0, 'Limits', [0, 2], ...
                      'Position', [65, y, 95, 24]);

% s replaces rho when CP is selected (same position, toggled via Visible)
lbl_s = uilabel(panel, 'Text', 's (dual):', ...
                'Position', [5, y, 55, 24], 'Visible', 'off');
ef_s  = uieditfield(panel, 'numeric', ...
                    'Value', 0.25, 'Limits', [0, Inf], ...
                    'Position', [65, y, 95, 24], 'Visible', 'off');

uilabel(panel, 'Text', 'maxiter:', 'Position', [180, y, 65, 24]);
ef_maxiter = uieditfield(panel, 'numeric', ...
                         'Value', 500, 'Limits', [1, 10000], ...
                         'RoundFractionalValues', 'on', ...
                         'Position', [250, y, 95, 24]);
y = y - 36;

% Row 3: tol
uilabel(panel, 'Text', 'tol:', 'Position', [5, y, 55, 24]);
ef_tol = uieditfield(panel, 'numeric', ...
                     'Value', 1e-4, 'Limits', [0, 1], ...
                     'Position', [65, y, 95, 24]);
y = y - 56;

% ===== Run button =====
btn_run = uibutton(panel, 'Text', 'Run Solver', ...
                   'Position', [5, y, 230, 42], ...
                   'BackgroundColor', [0.10, 0.60, 0.15], ...
                   'FontColor', 'white', 'FontWeight', 'bold', 'FontSize', 14, ...
                   'ButtonPushedFcn', @(~,~) cb_RunSolver(fig));
y = y - 52;

% ===== Status label =====
lbl_status = uilabel(panel, ...
    'Text', 'Ready. Load a blurred image and press Run Solver.', ...
    'Position', [5, y, PW, 22], 'FontAngle', 'italic');
y = y - 34;

% ===== PSNR / metrics =====
lbl_metrics = uilabel(panel, 'Text', '', ...
                      'Position', [5, y, PW, 26], ...
                      'FontWeight', 'bold', 'FontSize', 13);

% ---- Store handles ---------------------------------------------------------
app = fig.UserData;
app.dd_algorithm = dd_alg;
app.dd_problem   = dd_prob;
app.ef_gamma     = ef_gamma;
app.ef_t         = ef_t;
app.ef_rho       = ef_rho;
app.ef_s         = ef_s;
app.ef_maxiter   = ef_maxiter;
app.ef_tol       = ef_tol;
app.btn_run      = btn_run;
app.lbl_status   = lbl_status;
app.lbl_metrics  = lbl_metrics;
app.lbl_imgpath  = lbl_imgpath;
app.lbl_gtpath   = lbl_gtpath;
app.lbl_rho      = lbl_rho;
app.lbl_s        = lbl_s;
fig.UserData = app;

end


% =============================================================================
%  TAB 3 — ANALYSIS
%  Layout: [ax_original | ax_recon_analysis | ax_convergence | info panel]
% =============================================================================

function buildAnalysisTab(fig, tg)

tab = uitab(tg, 'Title', 'Analysis');

ax_orig = uiaxes(tab, 'Position', [10, 10, 258, 695]);
title(ax_orig, 'Ground Truth');
ax_orig.XTick = []; ax_orig.YTick = [];

ax_ra = uiaxes(tab, 'Position', [278, 10, 258, 695]);
title(ax_ra, 'Reconstruction');
ax_ra.XTick = []; ax_ra.YTick = [];

ax_conv = uiaxes(tab, 'Position', [546, 10, 258, 695]);
title(ax_conv, 'Objective vs Iteration');
xlabel(ax_conv, 'Iteration');
ylabel(ax_conv, 'Objective');

panel = uipanel(tab, 'Position', [814, 10, 576, 695]);

app = fig.UserData;
app.ax_original       = ax_orig;
app.ax_recon_analysis = ax_ra;
app.ax_convergence    = ax_conv;
fig.UserData = app;

buildAnalysisPanel(fig, panel);

end


function buildAnalysisPanel(fig, panel)

PW = 560;
y  = 655;

uilabel(panel, ...
    'Text', 'Run a solver in the Deblur tab, then switch here.', ...
    'Position', [5, y, PW, 20], 'FontAngle', 'italic');
y = y - 44;

% ===== Metrics =====
uilabel(panel, 'Text', 'Metrics', ...
        'Position', [5, y, PW, 20], 'FontWeight', 'bold');
y = y - 36;

lbl_psnr = uilabel(panel, 'Text', 'PSNR: —', ...
                   'Position', [5, y, PW, 32], ...
                   'FontSize', 16, 'FontWeight', 'bold');
y = y - 40;

lbl_iters = uilabel(panel, 'Text', 'Iterations: —', 'Position', [5, y, PW, 24]);
y = y - 30;
lbl_time  = uilabel(panel, 'Text', 'Time: —',       'Position', [5, y, PW, 24]);
y = y - 30;
lbl_conv  = uilabel(panel, 'Text', 'Converged: —',  'Position', [5, y, PW, 24]);
y = y - 52;

% ===== Refresh button =====
uibutton(panel, 'Text', 'Refresh Analysis', ...
         'Position', [5, y, 160, 30], ...
         'ButtonPushedFcn', @(~,~) updateAnalysisTab(fig));

% ---- Store handles ---------------------------------------------------------
app = fig.UserData;
app.lbl_psnr  = lbl_psnr;
app.lbl_iters = lbl_iters;
app.lbl_time  = lbl_time;
app.lbl_conv  = lbl_conv;
fig.UserData = app;

end


% =============================================================================
%  CALLBACKS — Corrupt tab
% =============================================================================

function cb_BrowseClean(fig)
% Browse for a clean image; display in ax_clean; clear stale results.
app = fig.UserData;

[fname, fpath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', 'Image files'}, ...
    'Select a clean image');
if isequal(fname, 0), return; end   % user cancelled dialog

try
    img = im2double(imread(fullfile(fpath, fname)));
    if size(img, 3) == 3
        img = rgb2gray(img);    % colour → grayscale
    end

    app.img_original  = img;
    app.imagePath     = fullfile(fpath, fname);
    app.img_blurred   = [];   % clear stale corrupted image
    app.img_recon     = [];   % clear stale reconstruction

    imshow(img, [], 'Parent', app.ax_clean);
    title(app.ax_clean, 'Clean Image');
    cla(app.ax_corrupted);
    title(app.ax_corrupted, 'Corrupted Image');

    app.lbl_imagepath_c.Text = fname;
    app.lbl_status_c.Text    = sprintf('Loaded %s  [%d×%d]', ...
                                        fname, size(img,1), size(img,2));
    fig.UserData = app;
catch ME
    uialert(fig, ME.message, 'Error loading image');
end

end


function cb_BlurTypeChanged(fig)
% Toggle Gaussian controls vs Motion controls in the Corrupt tab.
app = fig.UserData;

gauss = strcmp(app.dd_blur_c.Value, 'Gaussian');

app.lbl_ksize.Visible  = onoff(gauss);
app.ef_ksize.Visible   = onoff(gauss);
app.lbl_sigma.Visible  = onoff(gauss);
app.ef_sigma.Visible   = onoff(gauss);

app.lbl_mlen.Visible   = onoff(~gauss);
app.ef_mlen.Visible    = onoff(~gauss);
app.lbl_mangle.Visible = onoff(~gauss);
app.ef_mangle.Visible  = onoff(~gauss);

fig.UserData = app;

end


function cb_ApplyCorruption(fig)
% Build PSF, blur the clean image, add noise; display in ax_corrupted.
app = fig.UserData;

if isempty(app.img_original)
    uialert(fig, 'Browse for a clean image first.', 'No image loaded');
    return
end

try
    app.lbl_status_c.Text = 'Applying corruption...';
    drawnow;

    img = app.img_original;

    % Build the PSF from the chosen type
    switch app.dd_blur_c.Value
        case 'Gaussian'
            psf = gaussianPSF(app.ef_ksize.Value, app.ef_sigma.Value);
        case 'Motion'
            psf = motionPSF(app.ef_mlen.Value, app.ef_mangle.Value);
    end

    % FFT-based circular convolution — matches what the solvers assume,
    % and requires no Image Processing Toolbox.
    [H, W]   = size(img);
    [kH, kW] = size(psf);
    padded   = zeros(H, W);
    padded(1:kH, 1:kW) = psf;
    padded   = circshift(padded, -floor(kH/2), 1);
    padded   = circshift(padded, -floor(kW/2), 2);
    k_hat    = fft2(padded);
    b        = real(ifft2(k_hat .* fft2(img)));

    % Add noise (seed=42 for reproducibility)
    switch app.dd_noise_c.Value
        case 'Gaussian'
            b = addNoise(b, 'gaussian',    app.ef_nlevel.Value, 42);
        case 'Salt & Pepper'
            b = addNoise(b, 'saltpepper', app.ef_nlevel.Value, 42);
    end

    b = max(0, min(1, b));   % clip to [0,1] for safety

    app.img_blurred = b;
    app.psf         = psf;

    imshow(b, [], 'Parent', app.ax_corrupted);
    title(app.ax_corrupted, 'Corrupted Image');

    app.lbl_status_c.Text = sprintf('Done.  Blur=%s  Noise=%s (%.3f)', ...
        app.dd_blur_c.Value, app.dd_noise_c.Value, app.ef_nlevel.Value);

    % Mirror ground truth path label to Deblur tab (convenience)
    app.lbl_gtpath.Text = app.lbl_imagepath_c.Text;

    fig.UserData = app;
catch ME
    uialert(fig, ME.message, 'Corruption error');
    app.lbl_status_c.Text = ['Error: ' ME.message];
    fig.UserData = app;
end

end


function cb_UseCorrupted(fig)
% Copy b and psf into the Deblur tab (user doesn't need to re-load).
app = fig.UserData;

if isempty(app.img_blurred)
    uialert(fig, 'Apply corruption first.', 'No corrupted image');
    return
end

imshow(app.img_blurred, [], 'Parent', app.ax_blurred);
title(app.ax_blurred, 'Corrupted Input');

app.lbl_imgpath.Text = '(from Corrupt tab)';
app.lbl_status.Text  = 'Image ready from Corrupt tab.  Press Run Solver.';
fig.UserData = app;

end


% =============================================================================
%  CALLBACKS — Deblur tab
% =============================================================================

function cb_BrowseBlurred(fig)
% Load a blurred image directly into the Deblur tab.
app = fig.UserData;

[fname, fpath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', 'Image files'}, ...
    'Select blurred image');
if isequal(fname, 0), return; end

try
    img = im2double(imread(fullfile(fpath, fname)));
    if size(img, 3) == 3, img = rgb2gray(img); end

    app.img_blurred  = img;
    app.img_original = [];   % GT unknown when loading a blurred image directly
    app.psf          = [];   % PSF also unknown

    imshow(img, [], 'Parent', app.ax_blurred);
    title(app.ax_blurred, 'Corrupted Input');

    app.lbl_imgpath.Text = fname;
    app.lbl_status.Text  = sprintf('Loaded %s  [%d×%d]', fname, size(img,1), size(img,2));
    fig.UserData = app;
catch ME
    uialert(fig, ME.message, 'Error loading image');
end

end


function cb_BrowseGroundTruth(fig)
% Load a ground-truth image (used only for PSNR computation).
app = fig.UserData;

[fname, fpath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', 'Image files'}, ...
    'Select ground truth image');
if isequal(fname, 0), return; end

try
    img = im2double(imread(fullfile(fpath, fname)));
    if size(img, 3) == 3, img = rgb2gray(img); end
    app.img_original    = img;
    app.lbl_gtpath.Text = fname;
    fig.UserData = app;
catch ME
    uialert(fig, ME.message, 'Error loading ground truth');
end

end


function cb_AlgorithmChanged(fig)
% Swap visible parameter controls to match the selected algorithm.
app = fig.UserData;
val = app.dd_algorithm.Value;

switch val
    case 'CP'
        % CP has a separate dual step size s; rho is not used
        app.ef_rho.Visible = 'off'; app.lbl_rho.Visible = 'off';
        app.ef_s.Visible   = 'on';  app.lbl_s.Visible   = 'on';
        app.ef_t.Value     = 0.25;  % safe primal step for CP (t*s=0.0625 < 1/9)
        app.ef_s.Value     = 0.25;  % safe dual step
        app.ef_maxiter.Value = 500;
    case 'ADMM'
        app.ef_s.Visible   = 'off'; app.lbl_s.Visible   = 'off';
        app.ef_rho.Visible = 'on';  app.lbl_rho.Visible = 'on';
        app.ef_t.Value     = 1.0;
        app.ef_rho.Value   = 1.0;
        app.ef_maxiter.Value = 200; % ADMM converges faster, 200 is usually enough
    otherwise   % PDR or PDDR
        app.ef_s.Visible   = 'off'; app.lbl_s.Visible   = 'off';
        app.ef_rho.Visible = 'on';  app.lbl_rho.Visible = 'on';
        app.ef_t.Value     = 1.0;
        app.ef_rho.Value   = 1.0;
        app.ef_maxiter.Value = 500;
end

fig.UserData = app;

end


function cb_RunSolver(fig)
% Build config, run the chosen solver, display live reconstruction.
app = fig.UserData;

if app.solverRunning, return; end   % guard against double-click

if isempty(app.img_blurred)
    uialert(fig, 'Load a blurred image first (or use the Corrupt tab).', 'No image');
    return
end

% If PSF is unknown (user loaded a pre-blurred file), use a 1x1 identity kernel.
% The solver will then act purely as a denoiser.
if isempty(app.psf)
    app.psf = 1;
end

% --- Lock UI ---
app.solverRunning   = true;
app.btn_run.Enable  = 'off';
app.lbl_status.Text = 'Preparing solver...';
fig.UserData = app;
drawnow;

% --- Build config from UI controls ---
config.problem = lower(app.dd_problem.Value);  % 'l1' or 'l2'
config.gamma   = app.ef_gamma.Value;
config.t       = app.ef_t.Value;
config.rho     = app.ef_rho.Value;
config.s       = app.ef_s.Value;
config.maxiter = app.ef_maxiter.Value;
config.tol     = app.ef_tol.Value;
config.verbose = false;   % suppress console spam while app is running

% Live display: called from inside each solver's loop at every iteration.
% The callback definition lives here (one place); the invocation hooks are
% in each solver — this is the only way to get per-iteration updates in
% single-threaded MATLAB without a wrapper approach.
config.display_callback = @(x, k) liveUpdate(x, k, fig, config.maxiter);

% --- Run solver ---
try
    b   = app.img_blurred;
    psf = app.psf;

    switch app.dd_algorithm.Value
        case 'PDDR',  [x_sol, info] = PDDR(b, psf, config);
        case 'PDR',   [x_sol, info] = PDR(b, psf, config);
        case 'ADMM',  [x_sol, info] = ADMM(b, psf, config);
        case 'CP',    [x_sol, info] = CP(b, psf, config);
        otherwise,    error('Unknown algorithm: %s', app.dd_algorithm.Value);
    end

    % liveUpdate writes to fig.UserData each iteration — re-read after solver
    app = fig.UserData;
    app.img_recon = x_sol;
    app.info      = info;

    % Final reconstruction display
    imshow(x_sol, [], 'Parent', app.ax_recon);
    title(app.ax_recon, sprintf('Reconstruction — %s (%d iters)', ...
                                 app.dd_algorithm.Value, info.iterations));

    % PSNR (only if ground truth is loaded and size matches)
    if ~isempty(app.img_original) && isequal(size(app.img_original), size(x_sol))
        psnr_val = computePSNR(app.img_original, x_sol);
        app.lbl_metrics.Text = sprintf('PSNR = %.2f dB', psnr_val);
        app.lbl_status.Text  = sprintf('Done.  %d iters  |  %.2f s  |  PSNR = %.2f dB', ...
                                        info.iterations, info.time_sec, psnr_val);
    else
        app.lbl_metrics.Text = '';
        app.lbl_status.Text  = sprintf('Done.  %d iters  |  %.2f s', ...
                                        info.iterations, info.time_sec);
    end

    fig.UserData = app;
    updateAnalysisTab(fig);   % populate Analysis tab automatically

catch ME
    uialert(fig, ME.message, 'Solver error');
    app = fig.UserData;
    app.lbl_status.Text = ['Error: ' ME.message];
    fig.UserData = app;
end

% --- Unlock UI ---
app = fig.UserData;
app.solverRunning  = false;
app.btn_run.Enable = 'on';
fig.UserData = app;

end


% =============================================================================
%  HELPERS
% =============================================================================

function liveUpdate(x_cur, k, fig, maxiter)
% Refresh the reconstruction axes every iteration.
% Called from inside each solver's loop via config.display_callback.
% drawnow limitrate caps the refresh rate to ~20 Hz so drawing overhead
% doesn't dominate the solver's runtime.
app = fig.UserData;
imshow(x_cur, [], 'Parent', app.ax_recon);
app.lbl_status.Text = sprintf('Running...  iter %d / %d', k, maxiter);
fig.UserData = app;
drawnow limitrate;

end


function updateAnalysisTab(fig)
% Populate Tab 3 (Analysis) with the latest reconstruction and metrics.
app = fig.UserData;
if isempty(app.img_recon), return; end

% Show reconstruction
imshow(app.img_recon, [], 'Parent', app.ax_recon_analysis);
title(app.ax_recon_analysis, 'Reconstruction');

% Show ground truth if available
if ~isempty(app.img_original)
    imshow(app.img_original, [], 'Parent', app.ax_original);
    title(app.ax_original, 'Ground Truth');
end

% Convergence plot — all four solvers return info.objective
if ~isempty(app.info) && isfield(app.info, 'objective')
    cla(app.ax_convergence);
    plot(app.ax_convergence, app.info.objective, 'b-', 'LineWidth', 1.5);
    title(app.ax_convergence, 'Objective vs Iteration');
    xlabel(app.ax_convergence, 'Iteration');
    ylabel(app.ax_convergence, 'Objective value');
    grid(app.ax_convergence, 'on');
end

% Update metric labels
if ~isempty(app.info)
    info = app.info;

    if ~isempty(app.img_original) && isequal(size(app.img_original), size(app.img_recon))
        psnr_val = computePSNR(app.img_original, app.img_recon);
        app.lbl_psnr.Text = sprintf('PSNR: %.2f dB', psnr_val);
    else
        app.lbl_psnr.Text = 'PSNR: — (load ground truth)';
    end

    app.lbl_iters.Text = sprintf('Iterations: %d', info.iterations);
    app.lbl_time.Text  = sprintf('Time: %.2f s', info.time_sec);
    app.lbl_conv.Text  = sprintf('Converged: %s', ternary(info.converged, 'Yes', 'No'));
end

fig.UserData = app;

end


function s = onoff(flag)
% Convert logical to 'on' / 'off' string (for component Visible property).
if flag, s = 'on'; else, s = 'off'; end
end


function out = ternary(cond, a, b)
% Inline conditional: return a if cond, else b.
if cond, out = a; else, out = b; end
end
