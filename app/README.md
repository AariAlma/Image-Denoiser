# Image Deblurring Package — MATLAB

MATH 563 group project software package. Implements four proximal optimization
algorithms for image deblurring and denoising, a unified entry-point function,
a graphical user interface, a corruption pipeline, quality metrics, and a full
test suite.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Directory Structure](#directory-structure)
3. [Entry Point: `deblur()`](#entry-point-deblur)
4. [Algorithms](#algorithms)
5. [Graphical Interface: `DeblurApp`](#graphical-interface-deblurapp)
   - [Corrupt Tab](#corrupt-tab)
   - [Deblur Tab](#deblur-tab)
   - [Analysis Tab](#analysis-tab)
   - [Tune Tab](#tune-tab)
6. [Corruption Pipeline](#corruption-pipeline)
7. [Proximal Operators](#proximal-operators)
8. [Quality Metrics](#quality-metrics)
9. [Test Images](#test-images)
10. [Test Suite](#test-suite)

---

## Quick Start

```matlab
% Add paths (or run from project/app/ — paths are added automatically)
addpath('project/app')
addpath('project/app/algorithms')
addpath('project/app/prox_operators')

% Build a corrupted image
psf = gaussianPSF(9, 2.0);
b   = imfilter(im2double(imread('input_images/cameraman.jpg')), psf, 'circular');
b   = b + 0.02 * randn(size(b));

% Deblur with defaults
x = deblur('l2', 'PDDR', [], psf, b);

% Launch the GUI
DeblurApp
```

---

## Directory Structure

```
project/app/
├── deblur.m                  Entry-point function (teacher spec)
├── DeblurApp.m               GUI application
├── test_deblur.m             Test suite (~90 tests)
│
├── algorithms/
│   ├── PDR.m                 Primal Douglas-Rachford
│   ├── PDDR.m                Primal-Dual Douglas-Rachford
│   ├── ADMM.m                Alternating Direction Method of Multipliers
│   └── CP.m                  Chambolle-Pock
│
├── prox_operators/
│   ├── boxProx.m             Projection onto [0,1]
│   ├── l1Prox.m              Soft thresholding
│   ├── l1ShiftProx.m         Shifted L1 proximal
│   ├── l2prox.m              L2 Moreau scaling
│   ├── l2shiftprox.m         Shifted L2 proximal
│   ├── isoProx.m             Isotropic TV proximal
│   └── huberShiftProx.m      Huber loss proximal
│
├── metrics/
│   ├── computePSNR.m         Peak Signal-to-Noise Ratio
│   ├── computeSSIM.m         Structural Similarity Index
│   ├── computeMSE.m          Mean Squared Error
│   └── computeSNR.m          Signal-to-Noise Ratio
│
├── corrupting/
│   ├── buildCorruptedImage.m Blur + noise pipeline
│   ├── addNoise.m            Noise-only step
│   ├── gaussianPSF.m         Gaussian blur kernel
│   └── motionPSF.m           Motion blur kernel
│
├── input_images/             Test images (see §Test Images)
└── output_images/            Solver outputs are saved here
```

---

## Entry Point: `deblur()`

```matlab
x = deblur(problem, algorithm, xinitial, kernel, b)
x = deblur(problem, algorithm, xinitial, kernel, b, i)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `problem` | string | `'l1'`, `'l2'`, or `'huber'` — data fidelity term |
| `algorithm` | string | `'PDR'`, `'PDDR'`, `'ADMM'`, `'CP'` — case-insensitive |
| `xinitial` | H×W double or `[]` | Initial guess; `[]` starts from `b` |
| `kernel` | kH×kW double | Point-spread function (any size ≤ image) |
| `b` | H×W double | Blurred + noisy observation, values in [0,1] |
| `i` | struct (optional) | Parameter overrides — see table below |

### Parameter struct `i`

Any subset of these fields may be specified. Unset fields use algorithm defaults.

| Field | Default (PDR) | Default (ADMM) | Default (PDDR/CP) | Description |
|---|---|---|---|---|
| `gamma` | 0.006 (L1) / 0.012 (L2) | same | same | TV regularization weight |
| `t` | 0.25 | 1.0 | 1.0 | Primal step size |
| `rho` | 1.25 | 1.0 | 1.0 | Over-relaxation / ADMM penalty |
| `s` | — | — | 0.25 (CP only) | Dual step size (Chambolle-Pock) |
| `maxiter` | 500 | 200 | 500 | Maximum iterations |
| `tol` | 1e-4 | 1e-4 | 1e-4 | Convergence tolerance (relative change) |
| `delta` | 0.1 | 0.1 | 0.1 | Huber loss transition threshold |

### Output

`x` — H×W double, reconstructed image in [0,1].

A printed summary is always produced showing algorithm, iterations, objective value,
CPU time, and convergence status.

### Examples

```matlab
% Minimal call
x = deblur('l2', 'ADMM', [], psf, b);

% With warm start and custom gamma
i.gamma   = 0.008;
i.maxiter = 300;
x = deblur('l2', 'ADMM', x_prev, psf, b, i);

% L1 fidelity (salt-and-pepper noise)
x = deblur('l1', 'PDDR', [], psf, b_sp);

% Huber fidelity with custom delta
i.delta = 0.05;
x = deblur('huber', 'CP', [], psf, b, i);
```

---

## Algorithms

All four algorithms solve the same variational problem but differ in how they
split and handle the objective:

```
min_x  fidelity(Kx, b)  +  delta_[0,1](x)  +  gamma * ||Dx||_iso
```

where `K` is the blur operator (FFT-based circular convolution), `D` is the
discrete gradient, and `||·||_iso` is the isotropic total variation norm.

### PDR — Primal Douglas-Rachford

```matlab
[x_sol, info] = PDR(b, psf, config)
```

Handles both `f` and `g` in the primal. Solves the linear system
`(I + A^T A)^{-1}` per iteration via FFT. Maintains two envelope variables
`z1` (image space) and `z2` (range of A).

- Default step sizes: `t = 0.25`, `rho = 1.25`
- Convergence: relative change `||x_k - x_{k-1}|| / ||x_{k-1}|| < tol`

### PDDR — Primal-Dual Douglas-Rachford

```matlab
[x_sol, info] = PDDR(b, psf, config)
```

Introduces separate dual variables. Uses Moreau's identity to evaluate
`prox_{t·g*}` (conjugate proximal). Linear solve `(I + t^2 A^T A)^{-1}` via FFT.

- Default step sizes: `t = 1.0`, `rho = 1.0`
- Convergence: relative change OR objective plateau

### ADMM — Alternating Direction Method of Multipliers

```matlab
[x_sol, info] = ADMM(b, psf, config)
```

Splits into two equality constraints: `x = u` (box constraint) and `Ax = y`
(data fidelity + TV). Alternates between x-update (FFT linear solve),
u-update (box projection), y-update (data+TV proximal), and dual updates.

- Default: `t = 1.0`, `rho = 1.0`, `maxiter = 200`
- Convergence: all three residuals below `tol`:
  - `primal_res_u = ||x - u||`
  - `primal_res_y = ||Ax - y||`
  - `step_norm    = ||x_new - x_old||`

### CP — Chambolle-Pock

```matlab
[x_sol, info] = CP(b, psf, config)
```

Primal-Dual Hybrid Gradient. No linear system solve per iteration — uses
explicit gradient ascent on the dual then a proximal step on the primal.
Requires two step sizes satisfying `t · s · ||A||^2 ≤ 1`.

- Default step sizes: `t = 0.25`, `s = 0.25`
- `info` fields: `.objective`, `.rel_change`, `.iterations`, `.converged`, `.time_sec`

### `info` output struct (all algorithms)

| Field | Description |
|---|---|
| `info.iterations` | Number of iterations executed |
| `info.objective` | Objective value at each iteration |
| `info.converged` | `true` if tolerance was met before `maxiter` |
| `info.time_sec` | Wall-clock time in seconds |
| `info.rel_change` | (PDR/PDDR/CP) Relative iterate change per iteration |

---

## Graphical Interface: `DeblurApp`

Launch with:

```matlab
DeblurApp
```

The app is a 1400×750 px window with four mode tabs in the right sidebar.

---

### Corrupt Tab

Build a blurred and noisy observation from a clean image.

**Load Image**
- Dropdown `dd_clean` — select from `input_images/`
- Browse button — load any image from disk
- "Force grayscale" checkbox — converts colour images to grayscale

**Blur Kernel**
- Dropdown: `Gaussian` or `Motion`
- Gaussian: kernel size (odd, 1–99; default 9), sigma (default 2.0)
- Motion: length in pixels (default 15), angle in degrees (default 45°)

**Noise**
- Toggle between **Single** mode (one noise type) and **Multi** mode (up to three simultaneous noise sources)
- Supported noise types: Gaussian, Salt & Pepper, Speckle, Poisson, Uniform
- Noise level in [0, 1]

**Buttons**
- **Apply Corruption** — generates and displays the corrupted image
- **Use for Deblurring →** — passes the result and PSF to the Deblur tab

---

### Deblur Tab

Run a solver on a blurred image.

**Input**
- Dropdown — select a corrupted image (auto-populated from Corrupt tab)
- Browse button — load a custom blurred image from disk
- Ground-truth browse — load a reference image for PSNR/SSIM computation

**Algorithm & Problem**
- Solver: `PDDR` / `PDR` / `ADMM` / `CP`
- Problem: `L2` / `L1` / `Huber`

**Parameters** (auto-populated when algorithm or problem changes)

| Control | Visible for |
|---|---|
| Gamma (γ) | all |
| t | all |
| rho | PDR, PDDR, ADMM |
| s (dual step) | CP only |
| delta (δ) | Huber only |
| Iterations | all |
| Tolerance | all |

**Buttons**
- **Run Solver** (green) — executes `deblur()` and displays the result + PSNR
- **Run All Algorithms** (blue) — batch-runs all four solvers for comparison
- **Save Result...** (dark) — writes the reconstruction to `output_images/`

---

### Analysis Tab

Detailed quality metrics and convergence plots for the most recent solve.

**Quality Metrics**

| Metric | Description |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio (dB); higher is better |
| SSIM | Structural Similarity Index (0–1); higher is better |
| MSE | Mean Squared Error; lower is better |
| SNR | Signal-to-Noise Ratio (dB); higher is better |

**Solver Info**
- Iterations, wall time, convergence status

**Convergence Plots (2×2 grid)**
- Objective vs iteration
- PSNR vs iteration
- SSIM vs iteration
- MSE vs iteration

**Button**
- **Refresh Analysis** — regenerates all plots from the last solver run

---

### Tune Tab

Golden section search for the optimal regularization parameter.

**Setup**
- Algorithm and problem dropdowns (independent of Deblur tab)
- Metric to optimise: PSNR, SSIM, MSE, or SNR

**γ Search**
- Min / Max bounds (default 1e-4 to 2.0)
- Tolerance (default 0.05)
- **Run γ Search** (green) — bracket search; evaluates the solver at O(log(range/tol)) points
- **Apply γ* → Deblur** (blue) — copies optimal γ to the Deblur tab

**δ Search** (Huber problem only)
- Min / Max bounds (default 0.01 to 1.0)
- **Run delta Search** (orange)
- **Apply delta* → Deblur** (blue)

**Display**
- Left plot — evaluated (γ, metric) pairs during the search
- Right plot — reconstruction at the optimal parameter
- **Save γ* Image...** — save the optimal reconstruction

---

## Corruption Pipeline

### `buildCorruptedImage`

```matlab
[xTrue, b, psf] = buildCorruptedImage(imagePath)
[xTrue, b, psf] = buildCorruptedImage(imagePath, Name, Value, ...)
```

Complete pipeline: load → grayscale → blur → noise.

| Name-Value Pair | Default | Description |
|---|---|---|
| `BlurKind` | `'gaussian'` | `'gaussian'` or `'motion'` |
| `BlurSize` | `9` | Kernel size (Gaussian) |
| `BlurSigma` | `2.0` | Standard deviation (Gaussian) |
| `MotionLen` | `15` | Pixel span (motion) |
| `MotionAngle` | `45` | Direction in degrees (motion) |
| `NoiseType` | `'gaussian'` | See `addNoise` |
| `NoiseLevel` | `0.02` | Noise amplitude |
| `Seed` | `42` | RNG seed for reproducibility |

Returns: `xTrue` (clean grayscale double in [0,1]), `b` (blurred+noisy), `psf`.

### `addNoise`

```matlab
noisy = addNoise(img, noiseType, noiseLevel, seed)
```

| Noise Type | Model |
|---|---|
| `'gaussian'` | Additive white Gaussian: `b = x + σ·randn` |
| `'saltpepper'` | Random pixels set to 0 or 1 |
| `'speckle'` | Multiplicative: `b = x + x·σ·randn` |
| `'poisson'` | Photon-count model: `b = poissrnd(x·scale)/scale` |
| `'uniform'` | Additive uniform: `b = x + level·(2·rand - 1)` |

Output is clipped to [0,1].

### `gaussianPSF`

```matlab
psf = gaussianPSF(sz, sigma)
```

Returns a `sz × sz` normalized Gaussian kernel (sums to 1).

### `motionPSF`

```matlab
psf = motionPSF(len, angle)
```

Returns a `len × len` normalized motion-blur kernel. `angle = 0` is horizontal
right; increases counter-clockwise. Sums to 1.

---

## Proximal Operators

All operators are in `prox_operators/` and must be on the MATLAB path.

| Function | Signature | Formula |
|---|---|---|
| `boxProx` | `x = boxProx(v)` | `clip(v, 0, 1)` |
| `l1Prox` | `x = l1Prox(v, t)` | `sign(v) · max(|v| - t, 0)` |
| `l1ShiftProx` | `y = l1ShiftProx(v, b, t)` | `b + l1Prox(v - b, t)` |
| `l2prox` | `x = l2prox(v, t)` | `v / (2t + 1)` |
| `l2shiftprox` | `y = l2shiftprox(v, b, t)` | `b + l2prox(v - b, t)` |
| `isoProx` | `[u,v] = isoProx(a, b, t)` | Scale `(a,b)` by `max(1 - t/r, 0)` where `r = sqrt(a²+b²)` |
| `huberShiftProx` | `y = huberShiftProx(v, b, t, delta)` | L2 branch for `\|v-b\| ≤ (1+t)δ`, L1 branch otherwise |

---

## Quality Metrics

All metrics are in `metrics/` and take `(xTrue, xRecon)` as inputs (both H×W double in [0,1]).

| Function | Output | Formula | Better |
|---|---|---|---|
| `computePSNR` | dB | `10 log₁₀(1 / MSE)` | higher |
| `computeSSIM` | [0,1] | luminance · contrast · structure | higher |
| `computeMSE` | scalar | `mean((xTrue - xRecon).^2)` | lower |
| `computeSNR` | dB | `10 log₁₀(E[x²] / E[(x-xhat)²])` | higher |

`computeSSIM` supports both grayscale and colour images (returns mean across channels).

---

## Test Images

Located in `input_images/`:

| File | Description |
|---|---|
| `cameraman.jpg` | Standard 256×256 grayscale benchmark |
| `manWithHat.tiff` | Portrait test image |
| `mcgill.jpg` | Larger natural scene |
| `*Parque*` (5 files) | Same scene at 250px, 500px, 960px, 1920px, and original — used for resolution scaling studies |
| `*Tiburon*` | Underwater photograph |

---

## Test Suite

```matlab
cd project/app
test_deblur
```

Runs ~90 tests across 14 sections. Solver output is suppressed; only
`[PASS]` / `[FAIL]` lines are printed, with a summary table at the end.

| Section | What is tested |
|---|---|
| S1 | All 8 algorithm × problem combinations — size, range, finite, real |
| S2 | Algorithm string case insensitivity (`pdr`, `Pddr`, `aDmM`, …) |
| S3 | Problem string case insensitivity (`L1`, `L2`) |
| S4 | Error handling — unknown algorithm, numeric string, empty string |
| S5 | `xinitial` handling — `[]`, zeros, ones, warm start, determinism |
| S6 | Parameter struct overrides — maxiter, gamma, t, rho, s, unknown field |
| S7 | Default correctness — L1≠L2 gamma, PDR=0.25/1.25, ADMM maxiter=200 |
| S8 | Edge-case images — zeros, ones, uniform gray, boundary values, non-square |
| S9 | Edge-case PSFs — delta, 1×1 scalar, box, asymmetric 3×7, motion |
| S10 | Small images — 4×4, 8×8, 16×16 with all four algorithms |
| S11 | Huber problem — all algorithms + delta parameter effect |
| S12 | Extreme parameters — t=1e-6, t=100, gamma=1000, rho=1.99, CP divergent |
| S13 | Proximal operator unit tests — correctness of all 7 operators |
| S14 | Reconstruction quality — PSNR improves over corrupted input for all 8 combos |
