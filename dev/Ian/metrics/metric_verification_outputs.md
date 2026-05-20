## Image Quality Metrics

For the largest part, the quality of a given image is based on subjective perception. A lot of the time this is often biased, whether miss-informed or well-informed, by the specifications of the hardware of the equipment being used to take pictures. To test the mathematical or algorithmic superiority of our algorithms, one must design a hardware-agnostic quality metric, because “to-the-eye” is not a particularly good testament to the quality of two images whose qualitative difference is minute.

In this subfolder, I have proposed and designed a collection of metrics such that they are able to capture qualitative differences effectively. The main purpose of these metrics is to provide a quantified notion of reconstruction quality beyond subjective perception.

---

### 1. Peak Signal-to-Noise Ratio (PSNR)

PSNR is a natural extension of the classical Signal-to-Noise Ratio (SNR), a metric used abundantly in wireless communications and information theory. SNR measures the ratio between signal power and noise power, and is typically expressed in logarithmic scale using decibels (dB).

There is often some confusion as to whether the coefficient in front of the logarithm should be 10 or 20. This depends on whether one is measuring power directly or the amplitude of the signal itself. In image processing, PSNR is usually written in terms of the mean-squared error, and the “P” in PSNR stands for **Peak**, referring to the largest possible pixel value.

In this project, I use a pixel-regularized version of PSNR to measure the discrepancy between the true image and the reconstructed image.

#### Mathematical form

Using HTML formatting for compatibility with GitHub-style Markdown renderers, the PSNR may be written as

<p><b>PSNR(x<sub>true</sub>, x<sub>rec</sub>) = 10 log<sub>10</sub> ( MAX<sup>2</sup> / MSE )</b></p>

where

<p><b>MSE = (1 / N) &sum;<sub>i=1</sub><sup>N</sup> (x<sub>true,i</sub> - x<sub>rec,i</sub>)<sup>2</sup></b></p>

Here:
- <b>MAX</b> denotes the peak allowed pixel intensity
- <b>N</b> is the total number of pixels
- <b>MSE</b> is the mean-squared reconstruction error

#### Advantages
- Excellent for conveying information about **noise-intensity mismatch**
- Simple, classical, and easy to interpret
- Very effective when the dominant corruption is additive noise

#### Disadvantages
- It is not designed to capture quality mismatch due to **blurring**
- It is fundamentally a pointwise intensity comparison
- Hence it is not particularly useful as feedback on **deblurring capability**

Two images may have a similar PSNR while still looking visually very different if one is noticeably more blurred than the other.

---

### 2. Structural Similarity Index (SSIM)

SSIM is perhaps the closest one gets to quantifying the subjective perception of images. Unlike PSNR, it does not compare pixels only through raw error magnitude, but instead compares local image statistics in a way that is more aligned with how humans perceive visual structure.

Roughly speaking, SSIM compares:
- luminance
- contrast
- structural consistency

This makes SSIM much more sensitive to blur and structural degradation than PSNR.

#### Mathematical form

A standard local form of SSIM may be expressed as

<p><b>SSIM(x, y) = ((2&mu;<sub>x</sub>&mu;<sub>y</sub> + C<sub>1</sub>)(2&sigma;<sub>xy</sub> + C<sub>2</sub>)) / ((&mu;<sub>x</sub><sup>2</sup> + &mu;<sub>y</sub><sup>2</sup> + C<sub>1</sub>)(&sigma;<sub>x</sub><sup>2</sup> + &sigma;<sub>y</sub><sup>2</sup> + C<sub>2</sub>))</b></p>

where:
- <b>&mu;<sub>x</sub>, &mu;<sub>y</sub></b> are local means
- <b>&sigma;<sub>x</sub><sup>2</sup>, &sigma;<sub>y</sub><sup>2</sup></b> are local variances
- <b>&sigma;<sub>xy</sub></b> is the local covariance
- <b>C<sub>1</sub>, C<sub>2</sub></b> are small stabilizing constants

#### Advantages
- Captures **structural distortion**
- Sensitive to **blurring**
- More aligned with human visual judgment than PSNR

#### Disadvantages
- Still not a fully geometric metric
- Does not explicitly model the transport or displacement of intensity mass
- Can miss subtle spatial rearrangements when local statistics remain similar

Thus, SSIM is stronger than PSNR for perceptual assessment, but it is still not sufficient if one wishes to quantify the geometric relocation of image intensity.

---

### 3. Optimal Transport (OT)-Based Metric

To address the weaknesses of purely pointwise comparison metrics, I introduce an Optimal Transport based discrepancy.

The key principle is that image degradation, especially blurring, is not merely a matter of pixelwise error. Rather, it is often a matter of **mass redistribution**: intensity that was sharply localized in one region may be spread across neighboring regions.

Optimal Transport provides a mathematically principled way of quantifying exactly this phenomenon.

#### Intuition

Suppose one interprets the image intensity over a patch as a nonnegative mass distribution. Then the discrepancy between two patches is not merely how much mass differs at each pixel, but how much “effort” is required to move the mass in one patch so that it matches the mass in the other.

This makes OT especially suitable for:
- blur
- local shifts
- geometric redistribution of intensity
- shape mismatch

#### Basic transport formulation

Given two normalized nonnegative patches <b>a</b> and <b>b</b>, and a transport cost matrix <b>C</b>, the quadratic-cost transport problem may be written as

<p><b>W<sub>2</sub><sup>2</sup>(a, b) = min<sub>&pi; &isin; &Pi;(a,b)</sub> &sum;<sub>i,j</sub> C<sub>ij</sub> &pi;<sub>ij</sub></b></p>

where:
- <b>&pi;</b> is a transport plan
- <b>&Pi;(a,b)</b> denotes the set of couplings with marginals <b>a</b> and <b>b</b>
- <b>C<sub>ij</sub></b> is usually the squared Euclidean distance between pixel locations <b>i</b> and <b>j</b>

In words: we seek the least expensive way to rearrange the mass in one patch into the mass in the other.

---

### 3.1 Patchwise OT design in this project

A full global OT metric on the entire image is often too expensive and can also be too coarse to reflect local structural mismatch.
Therefore, one may proceed as follows -
1. Split the image into local patches
2. Normalize each patch into a probability mass distribution whenever possible
3. Compute a regularized transport discrepancy between corresponding patches
4. Aggregate these local scores into a global image discrepancy

This patchwise design provides
- locality
- computational tractability
- stronger sensitivity to local blur and shape distortion

---

### 3.2 Sinkhorn regularization
Wasserstein-2 IPM based Optimal Transport is pretty expensive and kind of an overkill.
Instead of solving the exact transport problem, we solve an entropically regularized version, which is much faster numerically and better suited for repeated patchwise evaluation.
This produces an approximation to the quadratic transport cost while remaining practical for image experiments.

---

### 3.3 Intensity-aware local discrepancy

In the code, the transport term is not used alone. A purely normalized OT distance can miss absolute intensity changes because two patches with different total masses may still have similar normalized shapes. Therefore, I augment the local discrepancy with:
- a **mass mismatch penalty**
- a **local L1 intensity mismatch**

The local discrepancy is therefore of the form

<p><b>Local Discrepancy = OT Shape Mismatch + &lambda;<sub>mass</sub> &middot; Relative Mass Mismatch + &lambda;<sub>L1</sub> &middot; Local Mean Absolute Error</b></p>

More explicitly, at the patch level this may be interpreted as

<p><b>d<sub>local</sub> = OT(p<sub>true</sub>, p<sub>rec</sub>) + &lambda;<sub>mass</sub> |m<sub>true</sub> - m<sub>rec</sub>| / max(m<sub>true</sub>, m<sub>rec</sub>, &epsilon;) + &lambda;<sub>L1</sub> mean(|p<sub>true</sub> - p<sub>rec</sub>|)</b></p>

where
- <b>p<sub>true</sub></b> and <b>p<sub>rec</sub></b> are corresponding true and reconstructed patches
- <b>m<sub>true</sub></b> and <b>m<sub>rec</sub></b> are their total masses
- <b>&lambda;<sub>mass</sub></b> and <b>&lambda;<sub>L1</sub></b> are tuning weights

This makes the metric sensitive to the magnitude of local intensity mismatch + the beneifts from the other two.
This is usefule here because -
- blurring is fundamentally a **redistribution** phenomenon
- local shifts in intensity should not always be penalized as harshly as pointwise metrics do
- the cost is geometrically meaningful

It therefore captures a notion of distortion which PSNR and SSIM only partially capture.

#### Advantages
- Captures **geometric displacement of intensity**
- Sensitive to **blur and mass spreading**
- More informative for reconstruction quality when structure is slightly displaced rather than destroyed
- Provides a physically meaningful discrepancy notion

#### Disadvantages
- More computationally expensive than PSNR and SSIM
- Requires tuning of:
  - patch size
  - stride
  - regularization strength
  - intensity penalty weights
- Sinkhorn regularization is approximate rather than exact

---

### 4. Composite Quality Metric

No single metric fully captures every relevant notion of image quality.

- PSNR is strong for raw intensity/noise mismatch
- SSIM is strong for perceptual structural similarity
- OT is strong for geometric mass-redistribution mismatch

Therefore, I combine them into a single holistic quality metric.

#### Composite form

The combined score is

<p><b>Q = w<sub>P</sub> &middot; Q<sub>PSNR</sub> + w<sub>S</sub> &middot; Q<sub>SSIM</sub> + w<sub>T</sub> &middot; Q<sub>OT</sub></b></p>

where:
- <b>w<sub>P</sub></b> is the PSNR weight
- <b>w<sub>S</sub></b> is the SSIM weight
- <b>w<sub>T</sub></b> is the OT weight

and the individual component scores are normalized to lie in <b>[0,1]</b>.

In the implementation, the overall score is then rescaled to <b>[0,100]</b> for interpretability.

---

### 4.1 Interpretation of the three terms

#### PSNR component
This term measures how well the reconstruction preserves raw pixel intensities and suppresses additive noise.

#### SSIM component
This term measures how well the reconstruction preserves the local structure and perceptual appearance of the image.

#### OT component
This term measures how well the reconstruction preserves the spatial organization of intensity mass.

Together, they provide a more balanced and robust notion of image quality than any one component alone.

---

### 4.2 Why a combined metric is necessary

A reconstruction may
- score highly in PSNR but still look blurry
- score well in SSIM while missing subtle mass shifts
- score well in OT shape while having noticeable intensity mismatch

Hence, relying on a single metric can be misleading. The combined metric is intended to overcome this by pooling complementary notions of fidelity.

---

### 5. Quality Labels

To make the final score more interpretable, I associate score intervals with qualitative labels.

| Score Range | Label |
|---|---|
| 90 - 100 | Close to SOTA |
| 75 - 90 | Excellent |
| 60 - 75 | Good |
| 40 - 60 | Satisfactory |
| Below 40 | Poor |

These labels are not universal scientific standards, but rather practical descriptors for summarizing performance.

---
### 6. Summary of Roles of the Metrics

| Metric | Main strength | Main weakness |
|---|---|---|
| PSNR | Excellent for measuring noise-intensity mismatch | Weak for blur and structural mismatch |
| SSIM | Good proxy for perceptual structural quality | Not fully geometric |
| OT-based discrepancy | Captures spatial redistribution and local shape mismatch | More computationally expensive |

This is just a complementary combination of the strengths of all.

### 7. Verification

In this section, I provide some simple verifications. One using the same image twice (this should produce very high similarity scores), and one using two completely different images (producing very high scores of mismatch).

---

### 7.1 - Images are alike

#### 7.1.1 - Display

<p align="center">
  <img src="https://github.com/user-attachments/assets/f213a50d-7c5d-4a6e-8728-da7ea831cff2" width="48%" />
  <img src="https://github.com/user-attachments/assets/0c9739a4-40c7-488d-9ecb-c0b6ca48d5ee" width="48%" />
</p>

---

#### 7.1.2 - Logs

```text
--- Raw Comparison ---
PSNR                      : 120.000000
SSIM                      : 1.000000
Transport discrepancy     : 0.000000
Mean local W2-shape       : 0.000092

--- Normalized Comparison Scores ---
PSNR score                : 0.997521
SSIM score                : 1.000000
Transport score           : 1.000000

--- Final ---
Quality score [0,100]     : 99.925637
Mismatch score [0,100]    : 0.074363
Quality label             : excellent
Mismatch label            : very small mismatch
```
---
### 7.2 Images are different

### 7.2.1 - Display
<p align="center">
<img width="1000" height="400" alt="cameraman_mcgill" src="https://github.com/user-attachments/assets/17542876-f7da-4b63-8e5a-e76bdedfac83" />
<img width="500" height="400" alt="cameraman_mcgill_information" src="https://github.com/user-attachments/assets/e213c866-8edd-4b96-8317-cfbd4e87deca" />
</p>

### 7.2.2 - Logs
```text
 Raw Comparison
PSNR                      : 9.171195
SSIM                      : 0.188037
Transport discrepancy     : 3.793717
Mean local W2-shape       : 1.548278

 Normalized Comparison Scores
PSNR score                : 0.367807
SSIM score                : 0.594019
Transport score           : 0.022512

 Final
Quality score [0,100]     : 32.612760
Mismatch score [0,100]    : 67.387240
Quality label             : very poor
Mismatch label            : very large mismatch

```


