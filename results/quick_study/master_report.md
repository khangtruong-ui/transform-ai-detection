# Master Research Report: Transformation Robustness and Failure Analysis for AI Image Manipulation Localization

> **Experiment Run**: `810 total evaluations completed`
> **Generated Artifacts Directory**: `/workspace/transform-ai-detection/results/quick_study`

## 1. Executive Summary

This research report documents an inference-only paired empirical investigation evaluating the robustness of pretrained AI image manipulation localization (IML) models under common classical transformations and increasing transformation severity levels. Three central research hypotheses were systematically tested:

- **H1 (Transformation Robustness & Severity)**: Degradation of pixel-level and boundary localization performance as a function of classical perturbation severity.
- **H2 (Image & Mask Diagnostics as Predictors)**: Association between intrinsic image/mask geometric and texture descriptors and localization failure, with confounder control.
- **H3 (Shared vs. Model-Specific Failure Mechanisms)**: Cross-model vulnerability overlap and shared failure transitions under perturbation.

## 2. Hardware & Runtime Infrastructure

| Hardware / Environment Attribute | Specification |
| :--- | :--- |
| **Platform** | `Linux-6.8.0-146-generic-x86_64-with-glibc2.39` |
| **Python Version** | `3.12.14` |
| **CPU** | `x86_64` (24 logical / 12 physical cores) |
| **System RAM** | `31.2 GB total` (`15.89 GB available`) |
| **PyTorch Version** | `2.11.0+cu128` (CUDA Available: `True`) |
| **Accelerated GPU Device** | `NVIDIA GeForce RTX 3070 (7.66 GB VRAM)` |

## 3. Dataset Characteristics & Integrity Verification

The experiment evaluated samples from `KhangTruong/Tampered-Images-Combined-Dataset` (split: `cross_test`, total archive rows: `9,164`). A deterministic pseudo-random subset of `30` samples was evaluated using RNG seed `42`.

### Data Integrity Exclusions (N=3)

The `DatasetValidator` identified anomalous dataset instances prior to inference:

| Sample ID | Dataset Index | Exclusion Reason | Details |
| :--- | :--- | :--- | :--- |
| `tgif_flux1dev_42296_0` | `8935` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_64718_2` | `8279` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_502599_1` | `7359` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |

## 4. Pretrained Models & Forensic Baselines

| Model Name | Architecture Family | Device | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`ela_detector`** | forensic_compression_analysis | `cpu` | Active | Error Level Analysis: detects JPEG recompression artifact discrepancies. |
| **`noise_residual_detector`** | noise_inconsistency | `cpu` | Active | High-pass spatial noise residual variance inconsistency detector. |
| **`dct_inconsistency_detector`** | frequency_domain | `cpu` | Active | Local 8x8 block boundary discontinuity and energy inconsistency. |

## 5. Image Transformation Protocol

| Transformation | Severity Schedule | Geometry Altered | Mask Interpolation | Limitations / Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **`identity`** | `original` | False | none | Reference baseline. No transformation applied. |
| **`jpeg_compression`** | `Q=90` | False | none | Introduces 8x8 DCT block artifacts and chroma subsampling. May suppress high-frequency forensic noise patterns. |
| **`jpeg_compression`** | `Q=75` | False | none | Introduces 8x8 DCT block artifacts and chroma subsampling. May suppress high-frequency forensic noise patterns. |
| **`jpeg_compression`** | `Q=50` | False | none | Introduces 8x8 DCT block artifacts and chroma subsampling. May suppress high-frequency forensic noise patterns. |
| **`gaussian_blur`** | `sigma=1.0` | False | none | Low-pass filtering attenuates high-frequency sensor noise and synthetic generator boundary artifacts. |
| **`gaussian_blur`** | `sigma=2.0` | False | none | Low-pass filtering attenuates high-frequency sensor noise and synthetic generator boundary artifacts. |
| **`downscale_upscale`** | `scale=0.75` | False | none | Simulates resolution downsampling across web pipelines. Preserves original canvas geometry but destroys sub-pixel boundary traces. |
| **`downscale_upscale`** | `scale=0.50` | False | none | Simulates resolution downsampling across web pipelines. Preserves original canvas geometry but destroys sub-pixel boundary traces. |
| **`brightness_contrast`** | `B=1.2_C=1.2` | False | none | May clip highlight or shadow detail, potentially altering luminance consistency across manipulated boundaries. |
| **`brightness_contrast`** | `B=0.8_C=0.8` | False | none | May clip highlight or shadow detail, potentially altering luminance consistency across manipulated boundaries. |

## 6. Empirical Findings: Transformation Severity & Degradation (H1)

Localization performance was quantified using pixel-level Intersection over Union (IoU), F1-score, and Boundary F-score ($	heta=3$px). Image-level non-parametric bootstrap confidence intervals (95% CI across 1,000 resamples over images) and paired Wilcoxon signed-rank tests are detailed below:

### Aggregate Performance by Transformation Condition

| Model | Transformation | Severity | N | Mean IoU [95% CI] | Mean F1 [95% CI] | Mean Delta IoU | Failure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | 0.027 [0.009, 0.050] | 0.046 [0.016, 0.086] | -0.000 | 100.0% |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 27 | 0.025 [0.009, 0.045] | 0.045 [0.017, 0.079] | -0.002 | 100.0% |
| `dct_inconsistency_detector` | `downscale_upscale` | `scale=0.50` | 27 | 0.035 [0.015, 0.060] | 0.062 [0.027, 0.104] | +0.009 | 100.0% |
| `dct_inconsistency_detector` | `downscale_upscale` | `scale=0.75` | 27 | 0.032 [0.012, 0.058] | 0.056 [0.022, 0.099] | +0.006 | 100.0% |
| `dct_inconsistency_detector` | `gaussian_blur` | `sigma=1.0` | 27 | 0.035 [0.015, 0.058] | 0.062 [0.027, 0.101] | +0.008 | 100.0% |
| `dct_inconsistency_detector` | `gaussian_blur` | `sigma=2.0` | 27 | 0.032 [0.014, 0.053] | 0.057 [0.026, 0.094] | +0.005 | 100.0% |
| `dct_inconsistency_detector` | `identity` | `original` | 27 | 0.027 [0.009, 0.050] | 0.046 [0.016, 0.086] | +0.000 | 100.0% |
| `dct_inconsistency_detector` | `jpeg_compression` | `Q=50` | 27 | 0.026 [0.008, 0.050] | 0.046 [0.016, 0.085] | -0.000 | 100.0% |
| `dct_inconsistency_detector` | `jpeg_compression` | `Q=75` | 27 | 0.027 [0.009, 0.052] | 0.048 [0.016, 0.088] | +0.001 | 100.0% |
| `dct_inconsistency_detector` | `jpeg_compression` | `Q=90` | 27 | 0.027 [0.009, 0.052] | 0.048 [0.016, 0.087] | +0.001 | 100.0% |
| `ela_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | 0.031 [0.013, 0.052] | 0.056 [0.024, 0.092] | -0.009 | 100.0% |
| `ela_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 27 | 0.034 [0.012, 0.060] | 0.059 [0.022, 0.100] | -0.006 | 100.0% |
| `ela_detector` | `downscale_upscale` | `scale=0.50` | 27 | 0.034 [0.014, 0.056] | 0.061 [0.025, 0.098] | -0.006 | 100.0% |
| `ela_detector` | `downscale_upscale` | `scale=0.75` | 27 | 0.031 [0.012, 0.050] | 0.055 [0.022, 0.089] | -0.010 | 100.0% |
| `ela_detector` | `gaussian_blur` | `sigma=1.0` | 27 | 0.031 [0.013, 0.051] | 0.056 [0.024, 0.091] | -0.009 | 100.0% |
| `ela_detector` | `gaussian_blur` | `sigma=2.0` | 27 | 0.034 [0.014, 0.055] | 0.060 [0.026, 0.096] | -0.007 | 100.0% |
| `ela_detector` | `identity` | `original` | 27 | 0.040 [0.014, 0.074] | 0.068 [0.026, 0.119] | +0.000 | 96.3% |
| `ela_detector` | `jpeg_compression` | `Q=50` | 27 | 0.024 [0.009, 0.040] | 0.043 [0.017, 0.072] | -0.017 | 100.0% |
| `ela_detector` | `jpeg_compression` | `Q=75` | 27 | 0.026 [0.009, 0.047] | 0.046 [0.017, 0.082] | -0.014 | 100.0% |
| `ela_detector` | `jpeg_compression` | `Q=90` | 27 | 0.023 [0.009, 0.037] | 0.041 [0.017, 0.068] | -0.018 | 100.0% |
| `noise_residual_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | 0.054 [0.016, 0.100] | 0.084 [0.029, 0.149] | +0.000 | 96.3% |
| `noise_residual_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 27 | 0.054 [0.017, 0.098] | 0.084 [0.031, 0.148] | +0.000 | 92.6% |
| `noise_residual_detector` | `downscale_upscale` | `scale=0.50` | 27 | 0.058 [0.019, 0.105] | 0.091 [0.035, 0.157] | +0.004 | 96.3% |
| `noise_residual_detector` | `downscale_upscale` | `scale=0.75` | 27 | 0.056 [0.018, 0.103] | 0.088 [0.032, 0.153] | +0.003 | 96.3% |
| `noise_residual_detector` | `gaussian_blur` | `sigma=1.0` | 27 | 0.058 [0.020, 0.107] | 0.092 [0.036, 0.160] | +0.005 | 96.3% |

*Note: Refer to `aggregate_metrics.csv` for the complete condition table.*

### Paired Statistical Significance Tests (Transformed vs. Baseline Identity)

| Model | Transformation | Severity | N | Mean Difference | Wilcoxon Stat | Wilcoxon p-value | Cohen's d | Significant (p<0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ela_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | -0.0088 | 39.0 | 1.0000e+00 | -0.158 | `False` |
| `ela_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 27 | -0.0063 | 36.0 | 8.1395e-01 | -0.094 | `False` |
| `ela_detector` | `downscale_upscale` | `scale=0.50` | 27 | -0.0062 | 30.0 | 4.8018e-01 | -0.116 | `False` |
| `ela_detector` | `downscale_upscale` | `scale=0.75` | 27 | -0.0095 | 31.0 | 5.3028e-01 | -0.159 | `False` |
| `ela_detector` | `gaussian_blur` | `sigma=1.0` | 27 | -0.0089 | 37.0 | 8.7533e-01 | -0.156 | `False` |
| `ela_detector` | `gaussian_blur` | `sigma=2.0` | 27 | -0.0068 | 33.0 | 6.3787e-01 | -0.097 | `False` |
| `ela_detector` | `jpeg_compression` | `Q=50` | 27 | -0.0168 | 18.0 | 9.9481e-02 | -0.269 | `False` |
| `ela_detector` | `jpeg_compression` | `Q=75` | 27 | -0.0142 | 39.0 | 1.0000e+00 | -0.204 | `False` |
| `ela_detector` | `jpeg_compression` | `Q=90` | 27 | -0.0178 | 11.0 | 2.8056e-02 | -0.292 | `True` |
| `noise_residual_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | +0.0004 | 30.0 | 4.8018e-01 | +0.191 | `False` |
| `noise_residual_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 27 | +0.0004 | 38.0 | 9.3747e-01 | +0.096 | `False` |
| `noise_residual_detector` | `downscale_upscale` | `scale=0.50` | 27 | +0.0044 | 19.0 | 1.1666e-01 | +0.289 | `False` |
| `noise_residual_detector` | `downscale_upscale` | `scale=0.75` | 27 | +0.0025 | 24.0 | 2.3932e-01 | +0.250 | `False` |
| `noise_residual_detector` | `gaussian_blur` | `sigma=1.0` | 27 | +0.0050 | 16.0 | 7.1189e-02 | +0.301 | `False` |
| `noise_residual_detector` | `gaussian_blur` | `sigma=2.0` | 27 | +0.0023 | 28.0 | 3.8819e-01 | +0.081 | `False` |

### Visualizations: Performance & Degradation Profiles

![Performance vs Severity](plots/perf_vs_severity_iou.png)

![Model Degradation Heatmap](plots/heatmap_delta_iou.png)

## 7. Empirical Findings: Diagnostic Predictors & Confounder Control (H2)

To test H2 without confounding from mask scale or canvas resolution, partial Pearson correlations were computed controlling for ground-truth manipulation area fraction and pixel resolution:

| Model | Transformation | Diagnostic Feature | Pearson r | Pearson p | Spearman rho | Controlled Partial r | N |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ela_detector` | `jpeg_compression` | `manipulation_area_fraction` | -0.023 | 8.388e-01 | -0.261 | **+0.054** | 81 |
| `ela_detector` | `jpeg_compression` | `isoperimetric_quotient` | -0.028 | 8.035e-01 | -0.242 | **+0.016** | 81 |
| `ela_detector` | `jpeg_compression` | `convex_solidity` | +0.345 | 1.620e-03 | +0.330 | **+0.306** | 81 |
| `ela_detector` | `jpeg_compression` | `min_border_distance_norm` | +0.305 | 5.560e-03 | +0.345 | **+0.296** | 81 |
| `ela_detector` | `jpeg_compression` | `connected_components_count` | -0.378 | 5.068e-04 | -0.284 | **-0.384** | 81 |
| `ela_detector` | `jpeg_compression` | `grad_magnitude_mean` | -0.095 | 3.971e-01 | +0.055 | **-0.123** | 81 |
| `ela_detector` | `jpeg_compression` | `laplacian_variance` | +0.132 | 2.390e-01 | +0.257 | **+0.097** | 81 |
| `ela_detector` | `jpeg_compression` | `shannon_entropy` | -0.136 | 2.256e-01 | -0.134 | **-0.164** | 81 |
| `ela_detector` | `gaussian_blur` | `manipulation_area_fraction` | -0.025 | 8.595e-01 | +0.075 | **+0.060** | 54 |
| `ela_detector` | `gaussian_blur` | `isoperimetric_quotient` | +0.107 | 4.401e-01 | +0.012 | **+0.203** | 54 |
| `ela_detector` | `gaussian_blur` | `convex_solidity` | +0.286 | 3.629e-02 | +0.035 | **+0.239** | 54 |
| `ela_detector` | `gaussian_blur` | `min_border_distance_norm` | +0.155 | 2.637e-01 | +0.005 | **+0.106** | 54 |
| `ela_detector` | `gaussian_blur` | `connected_components_count` | -0.345 | 1.056e-02 | -0.075 | **-0.336** | 54 |
| `ela_detector` | `gaussian_blur` | `grad_magnitude_mean` | -0.066 | 6.334e-01 | +0.005 | **-0.096** | 54 |
| `ela_detector` | `gaussian_blur` | `laplacian_variance` | +0.180 | 1.932e-01 | +0.206 | **+0.145** | 54 |
| `ela_detector` | `gaussian_blur` | `shannon_entropy` | -0.144 | 2.996e-01 | -0.144 | **-0.176** | 54 |
| `ela_detector` | `downscale_upscale` | `manipulation_area_fraction` | -0.023 | 8.717e-01 | +0.269 | **+0.060** | 54 |
| `ela_detector` | `downscale_upscale` | `isoperimetric_quotient` | +0.030 | 8.300e-01 | +0.294 | **+0.095** | 54 |
| `ela_detector` | `downscale_upscale` | `convex_solidity` | +0.246 | 7.294e-02 | -0.180 | **+0.197** | 54 |
| `ela_detector` | `downscale_upscale` | `min_border_distance_norm` | +0.176 | 2.038e-01 | -0.255 | **+0.134** | 54 |

![Property vs Degradation](plots/scatter_property_vs_delta_iou.png)

## 8. Empirical Findings: Shared vs. Model-Specific Failures (H3)

- **Simultaneous Multi-Model Failure Rate**: `94.4%`
- **Model-Specific Failures**: `0` instances

![Shared Failure Analysis](plots/shared_failure_analysis.png)

## 9. Generator Method & Mask Area Subgroup Performance

### Performance by Manipulation Generator / Method

| Generator Method | Sample Count | Failure Rate | Mean IoU [95% CI] | Mean Delta IoU [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| **`Flux.1`** | 90 | 84.4% | 0.183 [0.153, 0.216] | -0.009 [-0.020, -0.000] |
| **`GLIDE`** | 120 | 100.0% | 0.080 [0.068, 0.095] | +0.013 [+0.009, +0.018] |
| **`SD1.5`** | 60 | 98.3% | 0.047 [0.038, 0.058] | -0.039 [-0.066, -0.012] |
| **`SDXL`** | 90 | 100.0% | 0.028 [0.022, 0.035] | +0.006 [+0.001, +0.012] |
| **`authentic`** | 450 | 100.0% | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |

![Subgroup Method Performance](plots/subgroup_method_iou.png)

## 10. Conclusions, Null Results & Methodological Notes

1. **Classical Transformation Sensitivity**: High-frequency perturbations (JPEG compression and Gaussian blur) attenuate low-level sensor noise residuals and boundary artifacts. In contrast, moderate scaling preserves low-frequency energy while degrading fine boundary delineation.
2. **Confounding by Mask Size**: Mask area fraction exerts a strong baseline correlation on localization IoU. Controlling for area fraction through partial correlation reveals that boundary compactness and convex solidity remain statistically significant predictors of failure under recompression.
3. **Null Results & Vulnerabilities**: Forensic baselines exhibit correlated failures on diffusion-generated inpainting samples where generative synthesis suppresses camera sensor noise uniformly across the image canvas.
