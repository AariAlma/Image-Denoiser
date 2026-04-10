function DeblurApp()
% DeblurApp  MATLAB programmatic UI for MATH 563 image deblurring project.
%
% Layout: images fill the left ~1130px; a narrow right sidebar (~270px)
% holds mode-switch buttons, compact controls, and a scrolling status log.
%
% USAGE:
%   DeblurApp()

% ---- Paths ------------------------------------------------------------------
appDir = fileparts(mfilename('fullpath'));
if isempty(appDir), appDir = fileparts(which('DeblurApp')); end
addpath(fullfile(appDir,'algorithms'));
addpath(fullfile(appDir,'prox_operators'));
addpath(fullfile(appDir,'corrupting'));
addpath(fullfile(appDir,'metrics'));

% ---- Figure -----------------------------------------------------------------
fig = uifigure('Name','MATH 563 — Image Deblurring App', ...
               'Position',[50 50 1400 750]);

% ---- App state --------------------------------------------------------------
app.img_original  = [];      % H×W double, clean / ground truth
app.img_blurred   = [];      % H×W double, corrupted observation b
app.img_recon     = [];      % H×W double, latest solver output
app.psf           = [];      % blur kernel ([] = unknown)
app.info          = [];      % solver info struct from last run
app.imagePath     = '';
app.solverRunning = false;
app.appDir        = appDir;
app.currentMode   = 'corrupt';
% Per-iteration quality metric histories (filled by liveUpdate during solve)
app.iter_psnr     = [];
app.iter_mse      = [];
app.iter_ssim     = [];
app.iter_snr      = [];
app.currentChannel   = 0;   % 0=grayscale, 1-3=processing color channel c
app.recon_partial    = [];  % H×W×3 partial color recon during solve
app.img_deblur_input = [];  % image explicitly sent to Deblur tab
app.psf_deblur       = [];  % PSF for the deblur input ([] = identity)
app.inputDir         = fullfile(appDir, 'input_images');
app.outputDir        = fullfile(appDir, 'output_images');
app.cleanImageName   = '';  % filename of the currently loaded clean image
fig.UserData = app;

% ---- Two regions ------------------------------------------------------------
SBW = 270;   % sidebar width (px)
FW  = 1400;
FH  = 750;

mainArea = uipanel(fig, 'Position',[0, 0, FW-SBW, FH], 'BorderType','none');
sidebar  = uipanel(fig, 'Position',[FW-SBW, 0, SBW, FH], 'BorderType','line');

buildMainArea(fig, mainArea);
buildSidebar(fig, sidebar);

% ---- Start in Corrupt mode --------------------------------------------------
cb_SwitchMode(fig, 'corrupt');

end  % <<<< DeblurApp


% =============================================================================
%  MAIN AREA — all axes live here, shown/hidden by mode
% =============================================================================

function buildMainArea(fig, mainArea)
% mainArea is (FW-SBW) × FH = 1130 × 750 px.
% Each mode gets its own uipanel — toggling uipanel.Visible is reliable;
% toggling uiaxes.Visible directly is NOT (backgrounds bleed through).

AH = 720;   % axes height
AY = 15;    % bottom margin
MW = 1130;  % mainArea width

% -- One full-size panel per mode (same position, only one visible at a time) --
mp_corrupt  = uipanel(mainArea,'Position',[0,0,MW,750],'BorderType','none');
mp_deblur   = uipanel(mainArea,'Position',[0,0,MW,750],'BorderType','none','Visible','off');
mp_analysis = uipanel(mainArea,'Position',[0,0,MW,750],'BorderType','none','Visible','off');

% -- Corrupt mode axes (inside mp_corrupt) --
W2 = floor((MW - 20) / 2);   % ~555 px each
ax_clean     = uiaxes(mp_corrupt, 'Position',[5,     AY, W2, AH]);
ax_corrupted = uiaxes(mp_corrupt, 'Position',[W2+15, AY, W2, AH]);
title(ax_clean,'Clean Image');      ax_clean.XTick=[];     ax_clean.YTick=[];
title(ax_corrupted,'Corrupted Image'); ax_corrupted.XTick=[]; ax_corrupted.YTick=[];

% -- Deblur mode axes (inside mp_deblur) --
ax_blurred = uiaxes(mp_deblur, 'Position',[5,     AY, W2, AH]);
ax_recon   = uiaxes(mp_deblur, 'Position',[W2+15, AY, W2, AH]);
title(ax_blurred,'Corrupted Input');       ax_blurred.XTick=[]; ax_blurred.YTick=[];
title(ax_recon,  'Reconstruction (live)'); ax_recon.XTick=[];   ax_recon.YTick=[];

% -- Analysis mode axes (inside mp_analysis) -- 2×2 grid of convergence plots
% No images here; all four axes show a metric vs iteration.
W2a = floor((MW - 20) / 2);          % ~555 px per column
AHh = floor((750 - 45) / 2);         % ~352 px per row
AYb = 15;                             % bottom row y
AYt = AYb + AHh + 10;                % top row y

ax_obj  = uiaxes(mp_analysis, 'Position',[5,     AYt, W2a, AHh]);
ax_psnr = uiaxes(mp_analysis, 'Position',[W2a+15,AYt, W2a, AHh]);
ax_ssim = uiaxes(mp_analysis, 'Position',[5,     AYb, W2a, AHh]);
ax_mse  = uiaxes(mp_analysis, 'Position',[W2a+15,AYb, W2a, AHh]);

title(ax_obj, 'Objective vs Iteration');
xlabel(ax_obj,'Iteration'); ylabel(ax_obj,'Objective (log scale)'); grid(ax_obj,'on');

title(ax_psnr,'PSNR vs Iteration');
xlabel(ax_psnr,'Iteration'); ylabel(ax_psnr,'PSNR (dB)'); grid(ax_psnr,'on');

title(ax_ssim,'SSIM vs Iteration');
xlabel(ax_ssim,'Iteration'); ylabel(ax_ssim,'SSIM'); grid(ax_ssim,'on');

title(ax_mse, 'MSE vs Iteration');
xlabel(ax_mse,'Iteration'); ylabel(ax_mse,'MSE (log scale)'); grid(ax_mse,'on');

% Store handles
app = fig.UserData;
app.mp_corrupt    = mp_corrupt;
app.mp_deblur     = mp_deblur;
app.mp_analysis   = mp_analysis;
app.ax_clean      = ax_clean;
app.ax_corrupted  = ax_corrupted;
app.ax_blurred    = ax_blurred;
app.ax_recon      = ax_recon;
app.ax_obj        = ax_obj;
app.ax_psnr_plot  = ax_psnr;
app.ax_ssim_plot  = ax_ssim;
app.ax_mse_plot   = ax_mse;
fig.UserData = app;

end


% =============================================================================
%  SIDEBAR — mode buttons, per-mode controls, status log
% =============================================================================

function buildSidebar(fig, sidebar)
% sidebar is SBW × FH = 270 × 750 px.
SBW = 270;
SH  = 750;

% ---- Mode buttons (top, 3 × 88 px) -----------------------------------------
BH = 28;
BY = SH - BH - 5;   % = 717

btn_corrupt  = uibutton(sidebar,'Text','Corrupt', ...
    'Position',[1,   BY, 88, BH], ...
    'ButtonPushedFcn', @(~,~) cb_SwitchMode(fig,'corrupt'));
btn_deblur   = uibutton(sidebar,'Text','Deblur', ...
    'Position',[91,  BY, 88, BH], ...
    'ButtonPushedFcn', @(~,~) cb_SwitchMode(fig,'deblur'));
btn_analysis = uibutton(sidebar,'Text','Analysis', ...
    'Position',[181, BY, 88, BH], ...
    'ButtonPushedFcn', @(~,~) cb_SwitchMode(fig,'analysis'));

% ---- Status / error log (bottom) --------------------------------------------
log_area = uitextarea(sidebar, ...
    'Position',[5, 5, SBW-10, 118], ...
    'Editable','off', 'FontSize',9, ...
    'Value',{'[Ready] App started.'});

% ---- Per-mode control sub-panels (middle, all same position) ----------------
% Vertical space: from y=128 (above log) to y=714 (below buttons) = 586 px
PP = [2, 128, SBW-4, BY-128-4];   % [l, b, w, h] ≈ [2, 128, 266, 585]

panel_corrupt  = uipanel(sidebar,'Position',PP,'BorderType','none');
panel_deblur   = uipanel(sidebar,'Position',PP,'BorderType','none','Visible','off');
panel_analysis = uipanel(sidebar,'Position',PP,'BorderType','none','Visible','off');

buildCorruptPanel(fig, panel_corrupt);
buildDeblurPanel(fig, panel_deblur);
buildAnalysisPanel(fig, panel_analysis);

% Store
app = fig.UserData;
app.btn_corrupt    = btn_corrupt;
app.btn_deblur     = btn_deblur;
app.btn_analysis   = btn_analysis;
app.log_area       = log_area;
app.panel_corrupt  = panel_corrupt;
app.panel_deblur   = panel_deblur;
app.panel_analysis = panel_analysis;
fig.UserData = app;

end


% =============================================================================
%  CORRUPT SUB-PANEL
% =============================================================================

function buildCorruptPanel(fig, panel)
% panel ≈ 266 × 585 px (border-less). Layout from top down.
PW = 256;
y  = 555;

% -- Load Image ---------------------------------------------------------------
uilabel(panel,'Text','Input Images','Position',[5,y,PW,18],'FontWeight','bold');
y = y - 26;
app_tmp  = fig.UserData;
imgFiles = getInputImages(app_tmp.inputDir);
ddItems  = [{'(select from input folder...)'}; imgFiles];
dd_clean = uidropdown(panel,'Items',ddItems,'Value',ddItems{1}, ...
    'Position',[5,y,172,24], ...
    'ValueChangedFcn',@(dd,~) cb_SelectInputClean(fig,dd));
uibutton(panel,'Text','Browse...','Position',[182,y,PW-182,24], ...
    'ButtonPushedFcn',@(~,~) cb_BrowseClean(fig));
y = y - 34;

% -- Color mode ---------------------------------------------------------------
chk_gray = uicheckbox(panel,'Text','Force grayscale', ...
    'Value',false,'Position',[5,y,PW,22]);
y = y - 32;

% -- Blur Kernel --------------------------------------------------------------
uilabel(panel,'Text','Blur Kernel','Position',[5,y,PW,18],'FontWeight','bold');
y = y - 30;
uilabel(panel,'Text','Type:','Position',[5,y,38,24]);
dd_blur = uidropdown(panel,'Items',{'Gaussian','Motion'},'Value','Gaussian', ...
    'Position',[46,y,PW-46,24], ...
    'ValueChangedFcn',@(~,~) cb_BlurTypeChanged(fig));
y = y - 34;

% Gaussian controls (row 1)
lbl_ksize = uilabel(panel,'Text','Size:','Position',[5,y,35,22]);
ef_ksize  = uieditfield(panel,'numeric','Value',9,'Limits',[1,99], ...
    'RoundFractionalValues','on','Position',[43,y,58,22]);
lbl_sigma = uilabel(panel,'Text','Sigma:','Position',[108,y,44,22]);
ef_sigma  = uieditfield(panel,'numeric','Value',2.0,'Limits',[0.1,100], ...
    'Position',[154,y,60,22]);

% Motion controls (overlay same row, hidden by default)
lbl_mlen  = uilabel(panel,'Text','Len:','Position',[5,y,35,22],'Visible','off');
ef_mlen   = uieditfield(panel,'numeric','Value',15,'Limits',[1,500], ...
    'RoundFractionalValues','on','Position',[43,y,58,22],'Visible','off');
lbl_mangle= uilabel(panel,'Text','Ang°:','Position',[108,y,42,22],'Visible','off');
ef_mangle = uieditfield(panel,'numeric','Value',45,'Limits',[-360,360], ...
    'Position',[152,y,62,22],'Visible','off');
y = y - 42;

% -- Noise --------------------------------------------------------------------
uilabel(panel,'Text','Noise','Position',[5,y,PW,18],'FontWeight','bold');
y = y - 30;
uilabel(panel,'Text','Type:','Position',[5,y,38,24]);
dd_noise = uidropdown(panel,'Items',{'Gaussian','Salt & Pepper'},'Value','Gaussian', ...
    'Position',[46,y,PW-46,24]);
y = y - 34;
uilabel(panel,'Text','Level:','Position',[5,y,40,22]);
ef_nlevel = uieditfield(panel,'numeric','Value',0.02,'Limits',[0,1], ...
    'Position',[48,y,80,22]);
y = y - 46;

% -- Buttons ------------------------------------------------------------------
uibutton(panel,'Text','Apply Corruption', ...
    'Position',[5,y,PW,32], ...
    'BackgroundColor',[0.18,0.50,0.82],'FontColor','white','FontWeight','bold', ...
    'ButtonPushedFcn',@(~,~) cb_ApplyCorruption(fig));
y = y - 40;

uibutton(panel,'Text','Use for Deblurring  >', ...
    'Position',[5,y,PW,26], ...
    'ButtonPushedFcn',@(~,~) cb_UseCorrupted(fig));

% Store
app = fig.UserData;
app.dd_input_c     = dd_clean;
app.chk_grayscale  = chk_gray;
app.dd_blur_c  = dd_blur;
app.dd_noise_c = dd_noise;
app.ef_ksize   = ef_ksize;
app.ef_sigma   = ef_sigma;
app.ef_mlen    = ef_mlen;
app.ef_mangle  = ef_mangle;
app.ef_nlevel  = ef_nlevel;
app.lbl_ksize  = lbl_ksize;
app.lbl_sigma  = lbl_sigma;
app.lbl_mlen   = lbl_mlen;
app.lbl_mangle = lbl_mangle;
fig.UserData = app;

end


% =============================================================================
%  DEBLUR SUB-PANEL
% =============================================================================

function buildDeblurPanel(fig, panel)
PW = 256;
y  = 555;

% -- Load Blurred -------------------------------------------------------------
uilabel(panel,'Text','Blurred Image','Position',[5,y,PW,18],'FontWeight','bold');
y = y - 26;
app_tmp  = fig.UserData;
imgFiles = getInputImages(app_tmp.inputDir);
ddItems  = [{'(use Corrupt tab or select below...)'}; imgFiles];
dd_blur_in = uidropdown(panel,'Items',ddItems,'Value',ddItems{1}, ...
    'Position',[5,y,172,24], ...
    'ValueChangedFcn',@(dd,~) cb_SelectInputBlurred(fig,dd));
uibutton(panel,'Text','Browse...','Position',[182,y,PW-182,24], ...
    'ButtonPushedFcn',@(~,~) cb_BrowseBlurred(fig));
y = y - 34;

% -- Ground Truth -------------------------------------------------------------
uilabel(panel,'Text','Ground Truth  (PSNR)','Position',[5,y,PW,18],'FontWeight','bold');
y = y - 30;
uibutton(panel,'Text','Browse GT...','Position',[5,y,90,24], ...
    'ButtonPushedFcn',@(~,~) cb_BrowseGroundTruth(fig));
lbl_gtpath = uilabel(panel,'Text','auto from Corrupt tab', ...
    'Position',[100,y,PW-100,24],'FontAngle','italic','WordWrap','on');
y = y - 40;

% -- Algorithm ----------------------------------------------------------------
uilabel(panel,'Text','Algorithm','Position',[5,y,PW,18],'FontWeight','bold');
y = y - 30;
uilabel(panel,'Text','Solver:','Position',[5,y,44,24]);
dd_alg = uidropdown(panel,'Items',{'PDDR','PDR','ADMM','CP'},'Value','PDDR', ...
    'Position',[52,y,88,24], ...
    'ValueChangedFcn',@(~,~) cb_AlgorithmChanged(fig));
uilabel(panel,'Text','Prob:','Position',[148,y,36,24]);
dd_prob = uidropdown(panel,'Items',{'L2','L1'},'Value','L2', ...
    'Position',[187,y,69,24]);
y = y - 38;

% -- Parameters ---------------------------------------------------------------
uilabel(panel,'Text','Parameters','Position',[5,y,PW,18],'FontWeight','bold');
y = y - 30;

% Row 1: gamma, t
uilabel(panel,'Text','gamma:','Position',[5,y,44,22]);
ef_gamma = uieditfield(panel,'numeric','Value',0.006,'Limits',[0,Inf], ...
    'Position',[52,y,70,22]);
uilabel(panel,'Text','t:','Position',[132,y,16,22]);
ef_t = uieditfield(panel,'numeric','Value',1.0,'Limits',[0,Inf], ...
    'Position',[151,y,70,22]);
y = y - 30;

% Row 2: rho/s, maxiter
lbl_rho = uilabel(panel,'Text','rho:','Position',[5,y,32,22]);
ef_rho  = uieditfield(panel,'numeric','Value',1.0,'Limits',[0,2], ...
    'Position',[40,y,70,22]);
lbl_s   = uilabel(panel,'Text','s:','Position',[5,y,16,22],'Visible','off');
ef_s    = uieditfield(panel,'numeric','Value',0.25,'Limits',[0,Inf], ...
    'Position',[24,y,70,22],'Visible','off');
uilabel(panel,'Text','iter:','Position',[125,y,32,22]);
ef_maxiter = uieditfield(panel,'numeric','Value',500,'Limits',[1,10000], ...
    'RoundFractionalValues','on','Position',[160,y,60,22]);
y = y - 30;

% Row 3: tol
uilabel(panel,'Text','tol:','Position',[5,y,28,22]);
ef_tol = uieditfield(panel,'numeric','Value',1e-4,'Limits',[0,1], ...
    'Position',[36,y,78,22]);
y = y - 44;

% -- Run button ---------------------------------------------------------------
btn_run = uibutton(panel,'Text','Run Solver', ...
    'Position',[5,y,PW,36], ...
    'BackgroundColor',[0.10,0.60,0.15],'FontColor','white', ...
    'FontWeight','bold','FontSize',13, ...
    'ButtonPushedFcn',@(~,~) cb_RunSolver(fig));
y = y - 40;

% -- PSNR display -------------------------------------------------------------
lbl_metrics = uilabel(panel,'Text','', ...
    'Position',[5,y,PW,24],'FontWeight','bold','FontSize',12);
y = y - 36;

% -- Save result --------------------------------------------------------------
uibutton(panel,'Text','Save Result...','Position',[5,y,PW,26], ...
    'BackgroundColor',[0.25,0.25,0.25],'FontColor','white', ...
    'ButtonPushedFcn',@(~,~) cb_SaveResult(fig));

% Store
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
app.lbl_metrics  = lbl_metrics;
app.dd_input_d   = dd_blur_in;
app.lbl_gtpath   = lbl_gtpath;
app.lbl_rho      = lbl_rho;
app.lbl_s        = lbl_s;
fig.UserData = app;

end


% =============================================================================
%  ANALYSIS SUB-PANEL
% =============================================================================

function buildAnalysisPanel(fig, panel)
PW = 256;
y  = 545;

% ---- Quality metrics (scalar final values) ----------------------------------
uilabel(panel,'Text','Image Quality Metrics','Position',[5,y,PW,18], ...
    'FontWeight','bold');
y = y - 28;

lbl_psnr = uilabel(panel,'Text','PSNR:  —', ...
    'Position',[5,y,PW,20],'FontSize',11,'FontWeight','bold');
y = y - 24;
lbl_ssim = uilabel(panel,'Text','SSIM:  —','Position',[5,y,PW,20],'FontSize',11);
y = y - 24;
lbl_mse  = uilabel(panel,'Text','MSE:   —','Position',[5,y,PW,20],'FontSize',11);
y = y - 24;
lbl_snr  = uilabel(panel,'Text','SNR:   —','Position',[5,y,PW,20],'FontSize',11);
y = y - 32;

% ---- Run info ---------------------------------------------------------------
uilabel(panel,'Text','Solver Info','Position',[5,y,PW,18],'FontWeight','bold');
y = y - 26;
lbl_iters = uilabel(panel,'Text','Iterations: —','Position',[5,y,PW,20]);
y = y - 24;
lbl_time  = uilabel(panel,'Text','Time: —','Position',[5,y,PW,20]);
y = y - 24;
lbl_conv  = uilabel(panel,'Text','Converged: —','Position',[5,y,PW,20]);
y = y - 36;

uibutton(panel,'Text','Refresh Analysis', ...
    'Position',[5,y,PW,28], ...
    'ButtonPushedFcn',@(~,~) updateAnalysisTab(fig));

app = fig.UserData;
app.lbl_psnr_a = lbl_psnr;   % _a suffix: analysis panel (distinct from Deblur panel)
app.lbl_ssim_a = lbl_ssim;
app.lbl_mse_a  = lbl_mse;
app.lbl_snr_a  = lbl_snr;
app.lbl_iters  = lbl_iters;
app.lbl_time   = lbl_time;
app.lbl_conv   = lbl_conv;
fig.UserData = app;

end


% =============================================================================
%  MODE SWITCHING
% =============================================================================

function cb_SwitchMode(fig, mode)
app = fig.UserData;
app.currentMode = mode;

% Button styles
ACTIVE_BG = [0.18, 0.45, 0.85];
ACTIVE_FG = [1.00, 1.00, 1.00];
IDLE_BG   = [0.94, 0.94, 0.94];
IDLE_FG   = [0.15, 0.15, 0.15];

modes  = {'corrupt','deblur','analysis'};
btns   = {app.btn_corrupt, app.btn_deblur, app.btn_analysis};
panels = {app.panel_corrupt, app.panel_deblur, app.panel_analysis};

for i = 1:3
    active = strcmp(mode, modes{i});
    btns{i}.BackgroundColor = ternary(active, ACTIVE_BG, IDLE_BG);
    btns{i}.FontColor       = ternary(active, ACTIVE_FG, IDLE_FG);
    panels{i}.Visible       = onoff(active);
end

% Mode panel visibility (toggle panels, not individual axes — much more reliable)
isC = strcmp(mode,'corrupt');
isD = strcmp(mode,'deblur');
isA = strcmp(mode,'analysis');

app.mp_corrupt.Visible  = onoff(isC);
app.mp_deblur.Visible   = onoff(isD);
app.mp_analysis.Visible = onoff(isA);

fig.UserData = app;

end


% =============================================================================
%  CALLBACKS — Corrupt
% =============================================================================

function cb_BrowseClean(fig)
app = fig.UserData;
[fname, fpath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff','Image files'}, ...
    'Select a clean image');
if isequal(fname,0), return; end

try
    [img, wasGray] = collapseIfGray(im2double(imread(fullfile(fpath,fname))));
    app.img_original   = img;
    app.imagePath      = fullfile(fpath,fname);
    app.img_blurred    = [];
    app.img_recon      = [];
    app.cleanImageName = fname;
    % Add to input dropdown if not already listed, then select it
    items = app.dd_input_c.Items;
    if ~any(strcmp(items, fname))
        app.dd_input_c.Items = [items; {fname}];
    end
    app.dd_input_c.Value = fname;
    fig.UserData = app;

    colorshow(app.ax_clean, img);
    title(app.ax_clean,'Clean Image');
    cla(app.ax_corrupted); title(app.ax_corrupted,'Corrupted Image');

    appendLog(fig, sprintf('Loaded %s  [%d×%d %s]', fname, size(img,1), size(img,2), ...
        ternary(size(img,3)==3,'color','grayscale')));
    if wasGray
        appendLog(fig, 'Auto-converted: grayscale stored as RGB — using single channel');
    end
catch ME
    uialert(fig, ME.message, 'Error loading image');
    appendLog(fig, ['Load error: ' ME.message]);
end
end


function cb_BlurTypeChanged(fig)
app = fig.UserData;
gauss = strcmp(app.dd_blur_c.Value,'Gaussian');
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
app = fig.UserData;
if isempty(app.img_original)
    uialert(fig,'Browse for a clean image first.','No image');
    return
end

try
    appendLog(fig,'Applying corruption...');
    img = app.img_original;

    % Grayscale conversion (if requested and image is color)
    if app.chk_grayscale.Value && size(img, 3) == 3
        img = rgb2gray(img);
        app.img_original = img;   % keep GT consistent with what we blur
        colorshow(app.ax_clean, img);
        title(app.ax_clean, 'Clean Image (grayscale)');
    end

    switch app.dd_blur_c.Value
        case 'Gaussian', psf = gaussianPSF(app.ef_ksize.Value, app.ef_sigma.Value);
        case 'Motion',   psf = motionPSF(app.ef_mlen.Value, app.ef_mangle.Value);
    end

    % FFT-based circular convolution — per-channel for color, single-pass for grayscale
    [H,W,C] = size(img);
    [kH,kW] = size(psf);
    padded  = zeros(H,W);
    padded(1:kH,1:kW) = psf;
    padded  = circshift(padded,-floor(kH/2),1);
    padded  = circshift(padded,-floor(kW/2),2);
    k_hat   = fft2(padded);
    if C == 1
        b = real(ifft2(k_hat .* fft2(img)));   % H×W (2D) — preserve grayscale shape
    else
        b = zeros(H,W,C);
        for c = 1:C
            b(:,:,c) = real(ifft2(k_hat .* fft2(img(:,:,c))));
        end
    end

    switch app.dd_noise_c.Value
        case 'Gaussian',      b = addNoise(b,'gaussian',    app.ef_nlevel.Value, 42);
        case 'Salt & Pepper', b = addNoise(b,'saltpepper', app.ef_nlevel.Value, 42);
    end
    b = max(0,min(1,b));

    app.img_blurred = b;
    app.psf         = psf;
    app.lbl_gtpath.Text = app.cleanImageName;   % mirror GT label to Deblur panel
    fig.UserData = app;

    colorshow(app.ax_corrupted, b);
    title(app.ax_corrupted,'Corrupted Image');

    appendLog(fig, sprintf('Corruption done. Blur=%s  Noise=%s (%.3f)  Mode=%s', ...
        app.dd_blur_c.Value, app.dd_noise_c.Value, app.ef_nlevel.Value, ...
        ternary(size(b,3)==3,'color','grayscale')));
catch ME
    uialert(fig, ME.message, 'Corruption error');
    appendLog(fig, ['Corruption error: ' ME.message]);
end
end


function cb_UseCorrupted(fig)
app = fig.UserData;
if isempty(app.img_blurred)
    uialert(fig,'Apply corruption first.','No corrupted image');
    return
end
app.img_deblur_input = app.img_blurred;
app.psf_deblur       = app.psf;
colorshow(app.ax_blurred, app.img_blurred);
title(app.ax_blurred,'Corrupted Input');
fig.UserData = app;
appendLog(fig,'Image sent to Deblur tab. Press Run Solver.');
end


% =============================================================================
%  CALLBACKS — Deblur
% =============================================================================

function cb_BrowseBlurred(fig)
app = fig.UserData;
[fname,fpath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff','Image files'}, ...
    'Select blurred image');
if isequal(fname,0), return; end

try
    [img, wasGray] = collapseIfGray(im2double(imread(fullfile(fpath,fname))));
    app.img_deblur_input = img;
    app.psf_deblur       = [];
    app.img_original     = [];
    % Add to input dropdown if not already listed, then select it
    items = app.dd_input_d.Items;
    if ~any(strcmp(items, fname))
        app.dd_input_d.Items = [items; {fname}];
    end
    app.dd_input_d.Value = fname;
    fig.UserData = app;
    colorshow(app.ax_blurred, img);
    title(app.ax_blurred,'Corrupted Input');
    appendLog(fig, sprintf('Loaded blurred: %s  [%d×%d %s]', fname, size(img,1), size(img,2), ...
        ternary(size(img,3)==3,'color','grayscale')));
    if wasGray
        appendLog(fig, 'Auto-converted: grayscale stored as RGB — using single channel');
    end
catch ME
    uialert(fig, ME.message, 'Error loading image');
end
end


function cb_BrowseGroundTruth(fig)
app = fig.UserData;
[fname,fpath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff','Image files'}, ...
    'Select ground truth image');
if isequal(fname,0), return; end

try
    [img, wasGray] = collapseIfGray(im2double(imread(fullfile(fpath,fname))));
    app.img_original   = img;
    app.lbl_gtpath.Text = fname;
    fig.UserData = app;
    appendLog(fig, sprintf('Ground truth loaded: %s  [%s]', fname, ...
        ternary(size(img,3)==3,'color','grayscale')));
    if wasGray
        appendLog(fig, 'Auto-converted: grayscale stored as RGB — using single channel');
    end
catch ME
    uialert(fig, ME.message, 'Error loading ground truth');
end
end


function cb_AlgorithmChanged(fig)
app = fig.UserData;
val = app.dd_algorithm.Value;
switch val
    case 'CP'
        app.ef_rho.Visible = 'off'; app.lbl_rho.Visible = 'off';
        app.ef_s.Visible   = 'on';  app.lbl_s.Visible   = 'on';
        app.ef_t.Value     = 0.25;
        app.ef_s.Value     = 0.25;
        app.ef_maxiter.Value = 500;
    case 'ADMM'
        app.ef_s.Visible   = 'off'; app.lbl_s.Visible   = 'off';
        app.ef_rho.Visible = 'on';  app.lbl_rho.Visible = 'on';
        app.ef_t.Value     = 1.0;
        app.ef_rho.Value   = 1.0;
        app.ef_maxiter.Value = 200;
    otherwise
        app.ef_s.Visible   = 'off'; app.lbl_s.Visible   = 'off';
        app.ef_rho.Visible = 'on';  app.lbl_rho.Visible = 'on';
        app.ef_t.Value     = 1.0;
        app.ef_rho.Value   = 1.0;
        app.ef_maxiter.Value = 500;
end
fig.UserData = app;
end


function cb_RunSolver(fig)
app = fig.UserData;
if app.solverRunning, return; end

if isempty(app.img_deblur_input)
    uialert(fig,'Send an image to the Deblur tab first (use "Use for Deblurring >" or Browse).','No image');
    return
end

% Re-assert paths
addpath(fullfile(app.appDir,'algorithms'));
addpath(fullfile(app.appDir,'prox_operators'));
addpath(fullfile(app.appDir,'metrics'));

app.solverRunning  = true;
app.btn_run.Enable = 'off';
% Pre-allocate per-iteration metric arrays (filled by liveUpdate)
maxiter_val    = app.ef_maxiter.Value;
app.iter_psnr  = nan(maxiter_val, 1);
app.iter_mse   = nan(maxiter_val, 1);
app.iter_ssim  = nan(maxiter_val, 1);
app.iter_snr   = nan(maxiter_val, 1);
fig.UserData = app;
appendLog(fig,'Starting solver...');

config.problem = lower(app.dd_problem.Value);
config.gamma   = app.ef_gamma.Value;
config.t       = app.ef_t.Value;
config.rho     = app.ef_rho.Value;
config.s       = app.ef_s.Value;
config.maxiter = app.ef_maxiter.Value;
config.tol     = app.ef_tol.Value;
config.verbose = false;
config.display_callback = @(x,k) safeLiveUpdate(x,k,fig,config.maxiter);

try
    b    = app.img_deblur_input;
    psf  = app.psf_deblur;
    if isempty(psf), psf = 1; end   % identity if PSF unknown (direct browse case)
    algo = app.dd_algorithm.Value;
    C    = size(b, 3);

    if C == 1
        [x_sol, info] = runOneChannel(b, psf, config, algo);
    else
        x_sol = zeros(size(b));
        info  = [];
        app.recon_partial = zeros(size(b));
        fig.UserData = app;
        chan_names = {'R','G','B'};
        for c = 1:C
            appendLog(fig, sprintf('Solving %s channel (%d/3)...', chan_names{c}, c));
            app = fig.UserData;
            app.currentChannel = c;
            fig.UserData = app;
            ch_cfg = config;
            ch_cfg.display_callback = @(x,k) safeLiveUpdate(x,k,fig,config.maxiter);
            [x_sol(:,:,c), info_c] = runOneChannel(b(:,:,c), psf, ch_cfg, algo);
            app = fig.UserData;
            app.recon_partial(:,:,c) = x_sol(:,:,c);
            if isempty(info), info = info_c; end
            fig.UserData = app;
        end
        app = fig.UserData;
        app.currentChannel = 0;
        fig.UserData = app;
    end

    app = fig.UserData;
    app.img_recon = x_sol;
    app.info      = info;

    colorshow(app.ax_recon, x_sol);
    title(app.ax_recon, sprintf('Reconstruction — %s (%d iters)', algo, info.iterations));

    if ~isempty(app.img_original) && ...
            size(app.img_original,1)==size(x_sol,1) && ...
            size(app.img_original,2)==size(x_sol,2)
        psnr_val = computePSNR(app.img_original, x_sol);
        app.lbl_metrics.Text = sprintf('PSNR = %.2f dB', psnr_val);
        appendLog(fig, sprintf('Done. %d iters  %.2fs  PSNR=%.2fdB', ...
                               info.iterations, info.time_sec, psnr_val));
    else
        app.lbl_metrics.Text = '';
        appendLog(fig, sprintf('Done. %d iters  %.2fs', info.iterations, info.time_sec));
    end

    fig.UserData = app;
    updateAnalysisTab(fig);
catch ME
    uialert(fig, ME.message, 'Solver error');
    appendLog(fig, ['Solver error: ' ME.message]);
end

app = fig.UserData;
app.solverRunning  = false;
app.btn_run.Enable = 'on';
fig.UserData = app;
end


% =============================================================================
%  HELPERS
% =============================================================================

function safeLiveUpdate(x_cur, k, fig, maxiter)
% Wraps liveUpdate in try-catch so display errors never abort the solver.
try
    liveUpdate(x_cur, k, fig, maxiter);
catch
    try
        % Fallback: update just the log first line
        app = fig.UserData;
        existing = app.log_area.Value;
        line = sprintf('[iter %d/%d] Running...', k, maxiter);
        app.log_area.Value = [{line}; existing(2:min(end,30))];
        drawnow limitrate;
    catch
    end
end
end


function liveUpdate(x_cur, k, fig, maxiter)
% Refresh reconstruction display and record per-iteration quality metrics.
app = fig.UserData;

% Assemble display image — for color, stitch current channel into partial result
if app.currentChannel > 0
    app.recon_partial(:,:,app.currentChannel) = x_cur;
    displayImg = app.recon_partial;
    fig.UserData = app;
else
    displayImg = x_cur;
end
colorshow(app.ax_recon, displayImg);

% Update only the first line of the log (no flooding)
existing = app.log_area.Value;
if app.currentChannel > 0
    chan_names = {'R','G','B'};
    line = sprintf('[iter %d/%d] %s channel...', k, maxiter, chan_names{app.currentChannel});
else
    line = sprintf('[iter %d/%d] Running...', k, maxiter);
end
if isempty(existing)
    app.log_area.Value = {line};
else
    app.log_area.Value = [{line}; existing(2:min(end,30))];
end

% Record per-iteration quality metrics (grayscale only — color runs 3× sequentially).
% Requires ground truth with matching spatial AND channel dimensions.
if app.currentChannel == 0 && ~isempty(app.img_original) && ...
        size(app.img_original,1)==size(x_cur,1) && ...
        size(app.img_original,2)==size(x_cur,2) && ...
        size(app.img_original,3)==size(x_cur,3) && ...
        isfield(app,'iter_psnr') && k <= numel(app.iter_psnr)
    gt = app.img_original;
    app.iter_psnr(k) = computePSNR(gt, x_cur);
    app.iter_mse(k)  = computeMSE(gt,  x_cur);
    app.iter_ssim(k) = computeSSIM(gt, x_cur);
    app.iter_snr(k)  = computeSNR(gt,  x_cur);
    fig.UserData = app;
end

drawnow limitrate;
end


function updateAnalysisTab(fig)
app = fig.UserData;
if isempty(app.img_recon), return; end

% ---- Convergence plots (all use iteration as x-axis) -----------------------

% Objective vs iteration — log y-scale matches Aari's semilogy() convention
if ~isempty(app.info) && isfield(app.info,'objective')
    n = app.info.iterations;
    cla(app.ax_obj);
    semilogy(app.ax_obj, 1:n, app.info.objective(1:n), 'b-', 'LineWidth',1.5);
    title(app.ax_obj,'Objective vs Iteration');
    xlabel(app.ax_obj,'Iteration'); ylabel(app.ax_obj,'Objective (log scale)');
    grid(app.ax_obj,'on');
end

% PSNR / MSE / SSIM / SNR vs iteration (only when ground truth was available)
hasMetrics = ~isempty(app.iter_psnr) && any(~isnan(app.iter_psnr));

if hasMetrics
    n = find(~isnan(app.iter_psnr), 1, 'last');
    iters = 1:n;

    cla(app.ax_psnr_plot);
    plot(app.ax_psnr_plot, iters, app.iter_psnr(iters), 'r-', 'LineWidth',1.5);
    title(app.ax_psnr_plot,'PSNR vs Iteration');
    xlabel(app.ax_psnr_plot,'Iteration'); ylabel(app.ax_psnr_plot,'PSNR (dB)');
    grid(app.ax_psnr_plot,'on');

    cla(app.ax_ssim_plot);
    plot(app.ax_ssim_plot, iters, app.iter_ssim(iters), 'm-', 'LineWidth',1.5);
    title(app.ax_ssim_plot,'SSIM vs Iteration');
    xlabel(app.ax_ssim_plot,'Iteration'); ylabel(app.ax_ssim_plot,'SSIM');
    grid(app.ax_ssim_plot,'on');

    cla(app.ax_mse_plot);
    semilogy(app.ax_mse_plot, iters, app.iter_mse(iters), 'g-', 'LineWidth',1.5);
    title(app.ax_mse_plot,'MSE vs Iteration');
    xlabel(app.ax_mse_plot,'Iteration'); ylabel(app.ax_mse_plot,'MSE (log scale)');
    grid(app.ax_mse_plot,'on');
else
    % Determine why metrics are absent and show an informative message
    if ~isempty(app.img_original)
        noMetricsMsg = 'Per-iteration metrics disabled for color images';
    else
        noMetricsMsg = 'Load ground truth: use Corrupt tab or GT Browse';
    end
    for ax = {app.ax_psnr_plot, app.ax_ssim_plot, app.ax_mse_plot}
        cla(ax{1});
        text(ax{1}, 0.5, 0.5, noMetricsMsg, ...
             'Units','normalized','HorizontalAlignment','center', ...
             'FontSize',10,'Color',[0.5 0.5 0.5]);
    end
end

% ---- Sidebar scalar metrics -------------------------------------------------
if ~isempty(app.info)
    info = app.info;
    app.lbl_iters.Text = sprintf('Iterations: %d', info.iterations);
    app.lbl_time.Text  = sprintf('Time: %.2f s',   info.time_sec);
    app.lbl_conv.Text  = sprintf('Converged: %s',  ternary(info.converged,'Yes','No'));
end

if ~isempty(app.img_original) && ...
        size(app.img_original,1)==size(app.img_recon,1) && ...
        size(app.img_original,2)==size(app.img_recon,2) && ...
        size(app.img_original,3)==size(app.img_recon,3)
    gt = app.img_original;
    xr = app.img_recon;
    app.lbl_psnr_a.Text = sprintf('PSNR:  %.2f dB',  computePSNR(gt, xr));
    app.lbl_ssim_a.Text = sprintf('SSIM:  %.4f',     computeSSIM(gt, xr));
    app.lbl_mse_a.Text  = sprintf('MSE:   %.2e',     computeMSE(gt,  xr));
    app.lbl_snr_a.Text  = sprintf('SNR:   %.2f dB',  computeSNR(gt,  xr));
else
    app.lbl_psnr_a.Text = 'PSNR:  — (no ground truth)';
    app.lbl_ssim_a.Text = 'SSIM:  —';
    app.lbl_mse_a.Text  = 'MSE:   —';
    app.lbl_snr_a.Text  = 'SNR:   —';
end

fig.UserData = app;
end


function appendLog(fig, msg)
% Prepend a timestamped message to the status log (newest at top).
app = fig.UserData;
ts      = datestr(now,'HH:MM:SS');
newLine = sprintf('[%s] %s', ts, msg);
existing = app.log_area.Value;
if isempty(existing) || (numel(existing)==1 && isempty(char(existing{1})))
    app.log_area.Value = {newLine};
else
    combined = [{newLine}; existing(:)];
    app.log_area.Value = combined(1:min(end,30));  % keep last 30 lines
end
drawnow limitrate;
end


function colorshow(ax, img)
% Display a [0,1] image — handles both grayscale (H×W or H×W×1) and color (H×W×3).
img = max(0, min(1, img));
if size(img, 3) == 3
    image(ax, img);
else
    imagesc(ax, squeeze(img), [0 1]);   % squeeze H×W×1 → H×W before display
    colormap(ax, gray(256));
end
axis(ax, 'image');
axis(ax, 'off');
end


function s = onoff(flag)
% Logical → 'on' / 'off' string.
if flag, s = 'on'; else, s = 'off'; end
end


function out = ternary(cond, a, b)
% Inline conditional.
if cond, out = a; else, out = b; end
end


function [img, wasConverted] = collapseIfGray(img)
% If a 3-channel image has identical (or near-identical) channels,
% reduce it to single-channel grayscale.
% This handles grayscale images saved as RGB JPEGs (common with many tools).
% Threshold: 2/255 — tight enough that real color images are never collapsed.
% wasConverted is true when the auto-conversion fires.
wasConverted = false;
if size(img, 3) == 3
    maxDiff = max(max(max(abs(img(:,:,1) - img(:,:,2)))), ...
                  max(max(abs(img(:,:,1) - img(:,:,3)))));
    if maxDiff < 2/255
        img = img(:,:,1);
        wasConverted = true;
    end
end
end


function [x_ch, info_ch] = runOneChannel(b_ch, psf, config, algo)
% Dispatch a single-channel (H×W) solve to the appropriate solver.
switch algo
    case 'PDDR', [x_ch, info_ch] = PDDR(b_ch, psf, config);
    case 'PDR',  [x_ch, info_ch] = PDR(b_ch,  psf, config);
    case 'ADMM', [x_ch, info_ch] = ADMM(b_ch, psf, config);
    case 'CP',   [x_ch, info_ch] = CP(b_ch,   psf, config);
    otherwise,   error('Unknown algorithm: %s', algo);
end
end


function cb_SelectInputClean(fig, dd)
% Load the image selected in the Corrupt panel's input-folder dropdown.
val = dd.Value;
if isempty(val) || val(1) == '(', return; end   % ignore placeholder items
app = fig.UserData;
fpath = fullfile(app.inputDir, val);
try
    [img, wasGray] = collapseIfGray(im2double(imread(fpath)));
    app.img_original   = img;
    app.imagePath      = fpath;
    app.img_blurred    = [];
    app.img_recon      = [];
    app.cleanImageName = val;
    fig.UserData = app;
    colorshow(app.ax_clean, img);
    title(app.ax_clean, 'Clean Image');
    cla(app.ax_corrupted); title(app.ax_corrupted, 'Corrupted Image');
    appendLog(fig, sprintf('Loaded %s  [%d×%d %s]', val, size(img,1), size(img,2), ...
        ternary(size(img,3)==3,'color','grayscale')));
    if wasGray
        appendLog(fig, 'Auto-converted: grayscale stored as RGB — using single channel');
    end
catch ME
    uialert(fig, ME.message, 'Error loading image');
    appendLog(fig, ['Load error: ' ME.message]);
end
end


function cb_SelectInputBlurred(fig, dd)
% Load the image selected in the Deblur panel's input-folder dropdown.
val = dd.Value;
if isempty(val) || val(1) == '(', return; end   % ignore placeholder items
app = fig.UserData;
fpath = fullfile(app.inputDir, val);
try
    [img, wasGray] = collapseIfGray(im2double(imread(fpath)));
    app.img_deblur_input = img;
    app.psf_deblur       = [];
    app.img_original     = [];
    fig.UserData = app;
    colorshow(app.ax_blurred, img);
    title(app.ax_blurred, 'Corrupted Input');
    appendLog(fig, sprintf('Loaded blurred: %s  [%d×%d %s]', val, size(img,1), size(img,2), ...
        ternary(size(img,3)==3,'color','grayscale')));
    if wasGray
        appendLog(fig, 'Auto-converted: grayscale stored as RGB — using single channel');
    end
catch ME
    uialert(fig, ME.message, 'Error loading image');
    appendLog(fig, ['Load error: ' ME.message]);
end
end


function cb_SaveResult(fig)
% Save the current reconstruction to output_images/<input_name>/result_NNN.png.
% A subdirectory is created per input image; files are auto-incremented.
app = fig.UserData;
if isempty(app.img_recon)
    uialert(fig, 'Run the solver first — nothing to save yet.', 'No result');
    return
end

% Subdirectory name = input image base name (no extension)
if ~isempty(app.cleanImageName)
    [~, imgBase] = fileparts(app.cleanImageName);
elseif ~isempty(app.imagePath)
    [~, imgBase] = fileparts(app.imagePath);
else
    imgBase = 'unknown';
end

% Build per-image subdirectory inside output_images/
subDir = fullfile(app.outputDir, imgBase);
if ~isfolder(subDir)
    mkdir(subDir);
end

% Auto-increment: count existing result_NNN.png files
existing = dir(fullfile(subDir, 'result_*.png'));
n        = numel(existing) + 1;
defName  = sprintf('result_%03d.png', n);

% Save dialog pre-filled with the right folder and incremented name
[fname, fpath] = uiputfile( ...
    {'*.png','PNG image'; '*.tif;*.tiff','TIFF image'}, ...
    'Save reconstruction', fullfile(subDir, defName));
if isequal(fname, 0), return; end
imwrite(im2uint8(app.img_recon), fullfile(fpath, fname));
appendLog(fig, sprintf('Saved: output_images/%s/%s', imgBase, fname));
end


function files = getInputImages(inputDir)
% Return a sorted cell array of image filenames found in inputDir.
files = {};
if ~isfolder(inputDir), return; end
exts = {'*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff'};
for e = 1:numel(exts)
    d = dir(fullfile(inputDir, exts{e}));
    files = [files; {d.name}'];  %#ok<AGROW>
end
files = unique(files);
end
