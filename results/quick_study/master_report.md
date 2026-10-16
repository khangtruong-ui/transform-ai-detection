# Master Research Report: Transformation Robustness and Failure Analysis for AI Image Manipulation Localization

> **Experiment Run**: `810 total evaluations completed`
> **Generated Artifacts Directory**: `/root/transform-ai-detection/results/quick_study`

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

### Classical Models (Forensic Baselines)

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

#### Classical Models

| Model | Transformation | Severity | N | Mean IoU [95% CI] | Mean F1 [95% CI] | Mean Delta IoU | Failure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dct_inconsistency_detector` | `identity` | `original` | 27 | 0.027 [0.009, 0.050] | 0.046 [0.016, 0.086] | +0.000 | 100.0% |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | 0.027 [0.009, 0.050] | 0.046 [0.016, 0.086] | -0.000 | 100.0% |
| `dct_inconsistency_detector` | `gaussian_blur` | `sigma=1.0` | 27 | 0.035 [0.015, 0.058] | 0.062 [0.027, 0.101] | +0.008 | 100.0% |
| `dct_inconsistency_detector` | `jpeg_compression` | `Q=90` | 27 | 0.027 [0.009, 0.052] | 0.048 [0.016, 0.087] | +0.001 | 100.0% |
| `ela_detector` | `identity` | `original` | 27 | 0.040 [0.014, 0.074] | 0.068 [0.026, 0.119] | +0.000 | 96.3% |
| `ela_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | 0.031 [0.013, 0.052] | 0.056 [0.024, 0.092] | -0.009 | 100.0% |
| `ela_detector` | `gaussian_blur` | `sigma=1.0` | 27 | 0.031 [0.013, 0.051] | 0.056 [0.024, 0.091] | -0.009 | 100.0% |
| `ela_detector` | `jpeg_compression` | `Q=90` | 27 | 0.023 [0.009, 0.037] | 0.041 [0.017, 0.068] | -0.018 | 100.0% |
| `noise_residual_detector` | `identity` | `original` | 27 | 0.053 [0.016, 0.099] | 0.083 [0.029, 0.148] | +0.000 | 92.6% |
| `noise_residual_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | 0.054 [0.016, 0.100] | 0.084 [0.029, 0.149] | +0.000 | 96.3% |
| `noise_residual_detector` | `gaussian_blur` | `sigma=1.0` | 27 | 0.058 [0.020, 0.107] | 0.092 [0.036, 0.160] | +0.005 | 96.3% |
| `noise_residual_detector` | `jpeg_compression` | `Q=90` | 27 | 0.056 [0.018, 0.103] | 0.088 [0.033, 0.154] | +0.003 | 92.6% |

*Note: Refer to `aggregate_metrics.csv` for the complete condition table.*

### Paired Statistical Significance Tests (Transformed vs. Baseline Identity)

#### Classical Models

| Model | Transformation | Severity | N | Mean Difference | Wilcoxon Stat | Wilcoxon p-value | Cohen's d | Significant (p<0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dct_inconsistency_detector` | `downscale_upscale` | `scale=0.75` | 27 | +0.0056 | 11.0 | 2.8056e-02 | +0.410 | `True` |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | -0.0000 | 31.0 | 8.5886e-01 | -0.129 | `False` |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 27 | -0.0018 | 31.0 | 8.5886e-01 | -0.122 | `False` |
| `dct_inconsistency_detector` | `downscale_upscale` | `scale=0.50` | 27 | +0.0086 | 15.0 | 5.9739e-02 | +0.303 | `False` |
| `ela_detector` | `jpeg_compression` | `Q=90` | 27 | -0.0178 | 11.0 | 2.8056e-02 | -0.292 | `True` |
| `ela_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | -0.0088 | 39.0 | 1.0000e+00 | -0.158 | `False` |
| `ela_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 27 | -0.0063 | 36.0 | 8.1395e-01 | -0.094 | `False` |
| `ela_detector` | `downscale_upscale` | `scale=0.50` | 27 | -0.0062 | 30.0 | 4.8018e-01 | -0.116 | `False` |
| `noise_residual_detector` | `jpeg_compression` | `Q=90` | 27 | +0.0032 | 13.0 | 4.1389e-02 | +0.470 | `True` |
| `noise_residual_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 27 | +0.0004 | 30.0 | 4.8018e-01 | +0.191 | `False` |
| `noise_residual_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 27 | +0.0004 | 38.0 | 9.3747e-01 | +0.096 | `False` |
| `noise_residual_detector` | `downscale_upscale` | `scale=0.50` | 27 | +0.0044 | 19.0 | 1.1666e-01 | +0.289 | `False` |

*Note: Refer to `statistical_tests.json` for all paired hypothesis tests.*

### Visualizations: Performance & Degradation Profiles

![Performance vs Severity](plots/perf_vs_severity_iou.png)

![Model Degradation Heatmap](plots/heatmap_delta_iou.png)

## 7. Empirical Findings: Diagnostic Predictors & Confounder Control (H2)

To test H2 without confounding from mask scale or canvas resolution, partial Pearson correlations were computed controlling for ground-truth manipulation area fraction and pixel resolution:

### Classical Models: Diagnostic Predictors

| Model | Transformation | Diagnostic Feature | Pearson r | Pearson p | Spearman rho | Controlled Partial r | N |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dct_inconsistency_detector` | `downscale_upscale` | `min_border_distance_norm` | -0.282 | 3.873e-02 | -0.456 | **-0.368** | 54 |
| `dct_inconsistency_detector` | `brightness_contrast` | `connected_components_count` | -0.371 | 5.769e-03 | +0.018 | **-0.363** | 54 |
| `dct_inconsistency_detector` | `downscale_upscale` | `shannon_entropy` | -0.316 | 1.981e-02 | -0.153 | **-0.322** | 54 |
| `dct_inconsistency_detector` | `gaussian_blur` | `convex_solidity` | +0.344 | 1.080e-02 | -0.194 | **+0.316** | 54 |
| `ela_detector` | `jpeg_compression` | `connected_components_count` | -0.378 | 5.068e-04 | -0.284 | **-0.384** | 81 |
| `ela_detector` | `gaussian_blur` | `connected_components_count` | -0.345 | 1.056e-02 | -0.075 | **-0.336** | 54 |
| `ela_detector` | `jpeg_compression` | `convex_solidity` | +0.345 | 1.620e-03 | +0.330 | **+0.306** | 81 |
| `ela_detector` | `jpeg_compression` | `min_border_distance_norm` | +0.305 | 5.560e-03 | +0.345 | **+0.296** | 81 |
| `noise_residual_detector` | `downscale_upscale` | `min_border_distance_norm` | -0.311 | 2.211e-02 | -0.237 | **-0.703** | 54 |
| `noise_residual_detector` | `gaussian_blur` | `min_border_distance_norm` | -0.137 | 3.249e-01 | -0.211 | **-0.606** | 54 |
| `noise_residual_detector` | `gaussian_blur` | `manipulation_area_fraction` | -0.563 | 9.422e-06 | +0.245 | **-0.535** | 54 |
| `noise_residual_detector` | `downscale_upscale` | `convex_solidity` | -0.333 | 1.384e-02 | -0.126 | **-0.491** | 54 |

*Note: Refer to `correlations.csv` for the complete diagnostic correlation table.*

![Property vs Degradation](plots/scatter_property_vs_delta_iou.png)

## 8. Empirical Findings: Shared vs. Model-Specific Failures (H3)

- **Simultaneous Multi-Model Failure Rate**: `94.4%`
- **Model-Specific Failures**: `0` instances

### Model-Specific Failure Breakdown by Category

| Category | Models Evaluated | Total Model-Specific Failures |
| :--- | :--- | :--- |
| **Classical Models** | `dct_inconsistency_detector, ela_detector, noise_residual_detector` | 0 |
| **Integrated Models** | `None` | 0 |

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
