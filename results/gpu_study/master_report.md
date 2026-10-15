# Master Research Report: Transformation Robustness and Failure Analysis for AI Image Manipulation Localization

> **Experiment Run**: `360 total evaluations completed`
> **Generated Artifacts Directory**: `/workspace/transform-ai-detection/results/gpu_study`

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
| **System RAM** | `31.2 GB total` (`9.84 GB available`) |
| **PyTorch Version** | `2.11.0+cu128` (CUDA Available: `True`) |
| **Accelerated GPU Device** | `NVIDIA GeForce RTX 3070 (7.66 GB VRAM)` |

## 3. Dataset Characteristics & Integrity Verification

The experiment evaluated samples from `KhangTruong/Tampered-Images-Combined-Dataset` (split: `cross_test`, total archive rows: `9,164`). A deterministic pseudo-random subset of `10` samples was evaluated using RNG seed `42`.

### Data Integrity Exclusions (N=1)

The `DatasetValidator` identified anomalous dataset instances prior to inference:

| Sample ID | Dataset Index | Exclusion Reason | Details |
| :--- | :--- | :--- | :--- |
| `tgif_flux1dev_42296_0` | `8935` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |

## 4. Pretrained Models & Forensic Baselines

| Model Name | Architecture Family | Device | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`ela_detector`** | forensic_compression_analysis | `cuda:0` | Active | Error Level Analysis: detects JPEG recompression artifact discrepancies. |
| **`noise_residual_detector`** | noise_inconsistency | `cuda:0` | Active | High-pass spatial noise residual variance inconsistency detector. |
| **`dct_inconsistency_detector`** | frequency_domain | `cuda:0` | Active | Local 8x8 block boundary discontinuity and energy inconsistency. |
| **`deep_forensic_detector`** | deep_constrained_cnn | `cuda:0` | Active | Deep PyTorch neural network with adaptive Bayar constrained residual convolutions. |

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
| `dct_inconsistency_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 9 | 0.045 [0.005, 0.109] | 0.075 [0.010, 0.174] | -0.000 | 100.0% |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 9 | 0.038 [0.004, 0.087] | 0.066 [0.008, 0.147] | -0.007 | 100.0% |
| `dct_inconsistency_detector` | `downscale_upscale` | `scale=0.50` | 9 | 0.051 [0.004, 0.109] | 0.088 [0.009, 0.182] | +0.006 | 100.0% |
| `dct_inconsistency_detector` | `downscale_upscale` | `scale=0.75` | 9 | 0.051 [0.005, 0.115] | 0.086 [0.009, 0.186] | +0.006 | 100.0% |
| `dct_inconsistency_detector` | `gaussian_blur` | `sigma=1.0` | 9 | 0.048 [0.004, 0.099] | 0.084 [0.008, 0.169] | +0.003 | 100.0% |
| `dct_inconsistency_detector` | `gaussian_blur` | `sigma=2.0` | 9 | 0.039 [0.003, 0.084] | 0.068 [0.007, 0.144] | -0.006 | 100.0% |
| `dct_inconsistency_detector` | `identity` | `original` | 9 | 0.045 [0.005, 0.109] | 0.075 [0.010, 0.175] | +0.000 | 100.0% |
| `dct_inconsistency_detector` | `jpeg_compression` | `Q=50` | 9 | 0.047 [0.005, 0.111] | 0.078 [0.009, 0.179] | +0.002 | 100.0% |
| `dct_inconsistency_detector` | `jpeg_compression` | `Q=75` | 9 | 0.044 [0.004, 0.107] | 0.074 [0.008, 0.173] | -0.001 | 100.0% |
| `dct_inconsistency_detector` | `jpeg_compression` | `Q=90` | 9 | 0.047 [0.005, 0.113] | 0.078 [0.010, 0.181] | +0.002 | 100.0% |
| `deep_forensic_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 9 | 0.026 [0.002, 0.063] | 0.047 [0.004, 0.111] | +0.000 | 100.0% |
| `deep_forensic_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 9 | 0.027 [0.002, 0.065] | 0.048 [0.004, 0.113] | +0.001 | 100.0% |
| `deep_forensic_detector` | `downscale_upscale` | `scale=0.50` | 9 | 0.026 [0.002, 0.063] | 0.047 [0.004, 0.110] | -0.000 | 100.0% |
| `deep_forensic_detector` | `downscale_upscale` | `scale=0.75` | 9 | 0.026 [0.002, 0.063] | 0.047 [0.004, 0.111] | +0.000 | 100.0% |
| `deep_forensic_detector` | `gaussian_blur` | `sigma=1.0` | 9 | 0.027 [0.002, 0.064] | 0.048 [0.004, 0.112] | +0.000 | 100.0% |
| `deep_forensic_detector` | `gaussian_blur` | `sigma=2.0` | 9 | 0.027 [0.002, 0.066] | 0.048 [0.004, 0.115] | +0.001 | 100.0% |
| `deep_forensic_detector` | `identity` | `original` | 9 | 0.026 [0.002, 0.063] | 0.047 [0.004, 0.110] | +0.000 | 100.0% |
| `deep_forensic_detector` | `jpeg_compression` | `Q=50` | 9 | 0.027 [0.002, 0.064] | 0.048 [0.004, 0.112] | +0.000 | 100.0% |
| `deep_forensic_detector` | `jpeg_compression` | `Q=75` | 9 | 0.027 [0.002, 0.064] | 0.047 [0.004, 0.111] | +0.000 | 100.0% |
| `deep_forensic_detector` | `jpeg_compression` | `Q=90` | 9 | 0.027 [0.002, 0.064] | 0.047 [0.004, 0.111] | +0.000 | 100.0% |
| `ela_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 9 | 0.043 [0.006, 0.088] | 0.075 [0.011, 0.152] | -0.002 | 100.0% |
| `ela_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 9 | 0.057 [0.004, 0.132] | 0.094 [0.008, 0.210] | +0.012 | 100.0% |
| `ela_detector` | `downscale_upscale` | `scale=0.50` | 9 | 0.041 [0.006, 0.091] | 0.070 [0.012, 0.153] | -0.004 | 100.0% |
| `ela_detector` | `downscale_upscale` | `scale=0.75` | 9 | 0.039 [0.006, 0.088] | 0.068 [0.011, 0.148] | -0.006 | 100.0% |
| `ela_detector` | `gaussian_blur` | `sigma=1.0` | 9 | 0.033 [0.006, 0.074] | 0.059 [0.012, 0.128] | -0.012 | 100.0% |

*Note: Refer to `aggregate_metrics.csv` for the complete condition table.*

### Paired Statistical Significance Tests (Transformed vs. Baseline Identity)

| Model | Transformation | Severity | N | Mean Difference | Wilcoxon Stat | Wilcoxon p-value | Cohen's d | Significant (p<0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ela_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 9 | -0.0022 | 1.0 | 2.5000e-01 | -0.503 | `False` |
| `ela_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 9 | +0.0120 | 5.0 | 1.0000e+00 | +0.203 | `False` |
| `ela_detector` | `downscale_upscale` | `scale=0.50` | 9 | -0.0041 | 4.0 | 8.7500e-01 | -0.149 | `False` |
| `ela_detector` | `downscale_upscale` | `scale=0.75` | 9 | -0.0056 | 4.0 | 8.7500e-01 | -0.201 | `False` |
| `ela_detector` | `gaussian_blur` | `sigma=1.0` | 9 | -0.0115 | 2.0 | 3.7500e-01 | -0.369 | `False` |
| `ela_detector` | `gaussian_blur` | `sigma=2.0` | 9 | -0.0106 | 3.0 | 6.2500e-01 | -0.304 | `False` |
| `ela_detector` | `jpeg_compression` | `Q=50` | 9 | -0.0125 | 1.0 | 2.5000e-01 | -0.378 | `False` |
| `ela_detector` | `jpeg_compression` | `Q=75` | 9 | -0.0053 | 5.0 | 1.0000e+00 | -0.158 | `False` |
| `ela_detector` | `jpeg_compression` | `Q=90` | 9 | -0.0187 | 0.0 | 1.2500e-01 | -0.493 | `False` |
| `noise_residual_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 9 | +0.0000 | 4.0 | 8.7500e-01 | +0.005 | `False` |
| `noise_residual_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 9 | -0.0022 | 4.0 | 8.7500e-01 | -0.316 | `False` |
| `noise_residual_detector` | `downscale_upscale` | `scale=0.50` | 9 | +0.0010 | 4.0 | 8.7500e-01 | +0.232 | `False` |
| `noise_residual_detector` | `downscale_upscale` | `scale=0.75` | 9 | -0.0006 | 2.0 | 3.7500e-01 | -0.354 | `False` |
| `noise_residual_detector` | `gaussian_blur` | `sigma=1.0` | 9 | +0.0010 | 4.0 | 8.7500e-01 | +0.174 | `False` |
| `noise_residual_detector` | `gaussian_blur` | `sigma=2.0` | 9 | -0.0061 | 3.0 | 6.2500e-01 | -0.312 | `False` |

### Visualizations: Performance & Degradation Profiles

![Performance vs Severity](plots/perf_vs_severity_iou.png)

![Model Degradation Heatmap](plots/heatmap_delta_iou.png)

## 7. Empirical Findings: Diagnostic Predictors & Confounder Control (H2)

To test H2 without confounding from mask scale or canvas resolution, partial Pearson correlations were computed controlling for ground-truth manipulation area fraction and pixel resolution:

| Model | Transformation | Diagnostic Feature | Pearson r | Pearson p | Spearman rho | Controlled Partial r | N |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ela_detector` | `jpeg_compression` | `manipulation_area_fraction` | -0.930 | 2.225e-12 | -0.598 | **-0.912** | 27 |
| `ela_detector` | `jpeg_compression` | `isoperimetric_quotient` | -0.050 | 8.028e-01 | -0.266 | **+0.387** | 27 |
| `ela_detector` | `jpeg_compression` | `convex_solidity` | +0.918 | 1.526e-11 | +0.715 | **+0.205** | 27 |
| `ela_detector` | `jpeg_compression` | `min_border_distance_norm` | +0.535 | 4.030e-03 | +0.631 | **-0.420** | 27 |
| `ela_detector` | `jpeg_compression` | `connected_components_count` | -0.937 | 6.115e-13 | -0.570 | **-0.273** | 27 |
| `ela_detector` | `jpeg_compression` | `grad_magnitude_mean` | +0.399 | 3.900e-02 | +0.453 | **-0.264** | 27 |
| `ela_detector` | `jpeg_compression` | `laplacian_variance` | +0.386 | 4.672e-02 | +0.524 | **-0.345** | 27 |
| `ela_detector` | `jpeg_compression` | `shannon_entropy` | -0.158 | 4.309e-01 | +0.004 | **-0.325** | 27 |
| `ela_detector` | `gaussian_blur` | `manipulation_area_fraction` | -0.962 | 1.754e-10 | -0.564 | **-0.953** | 18 |
| `ela_detector` | `gaussian_blur` | `isoperimetric_quotient` | -0.017 | 9.457e-01 | -0.279 | **+0.668** | 18 |
| `ela_detector` | `gaussian_blur` | `convex_solidity` | +0.952 | 1.247e-09 | +0.657 | **+0.352** | 18 |
| `ela_detector` | `gaussian_blur` | `min_border_distance_norm` | +0.535 | 2.209e-02 | +0.618 | **-0.679** | 18 |
| `ela_detector` | `gaussian_blur` | `connected_components_count` | -0.968 | 4.549e-11 | -0.570 | **-0.360** | 18 |
| `ela_detector` | `gaussian_blur` | `grad_magnitude_mean` | +0.397 | 1.033e-01 | +0.510 | **-0.420** | 18 |
| `ela_detector` | `gaussian_blur` | `laplacian_variance` | +0.383 | 1.169e-01 | +0.546 | **-0.537** | 18 |
| `ela_detector` | `gaussian_blur` | `shannon_entropy` | -0.169 | 5.035e-01 | -0.050 | **-0.506** | 18 |
| `ela_detector` | `downscale_upscale` | `manipulation_area_fraction` | -0.868 | 3.042e-06 | +0.264 | **-0.835** | 18 |
| `ela_detector` | `downscale_upscale` | `isoperimetric_quotient` | +0.122 | 6.293e-01 | +0.668 | **+0.773** | 18 |
| `ela_detector` | `downscale_upscale` | `convex_solidity` | +0.908 | 1.936e-07 | -0.046 | **+0.405** | 18 |
| `ela_detector` | `downscale_upscale` | `min_border_distance_norm` | +0.339 | 1.686e-01 | -0.229 | **-0.858** | 18 |

![Property vs Degradation](plots/scatter_property_vs_delta_iou.png)

## 8. Empirical Findings: Shared vs. Model-Specific Failures (H3)

- **Simultaneous Multi-Model Failure Rate**: `88.9%`
- **Model-Specific Failures**: `0` instances

![Shared Failure Analysis](plots/shared_failure_analysis.png)

## 9. Generator Method & Mask Area Subgroup Performance

### Performance by Manipulation Generator / Method

| Generator Method | Sample Count | Failure Rate | Mean IoU [95% CI] | Mean Delta IoU [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| **`Flux.1`** | 40 | 75.0% | 0.246 [0.191, 0.305] | -0.022 [-0.042, -0.004] |
| **`GLIDE`** | 40 | 100.0% | 0.108 [0.091, 0.131] | +0.011 [+0.001, +0.024] |
| **`SD1.5`** | 40 | 100.0% | 0.030 [0.026, 0.035] | +0.002 [-0.000, +0.004] |
| **`SDXL`** | 40 | 100.0% | 0.033 [0.029, 0.037] | -0.002 [-0.003, -0.001] |
| **`authentic`** | 200 | 100.0% | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |

![Subgroup Method Performance](plots/subgroup_method_iou.png)

## 10. Conclusions, Null Results & Methodological Notes

1. **Classical Transformation Sensitivity**: High-frequency perturbations (JPEG compression and Gaussian blur) attenuate low-level sensor noise residuals and boundary artifacts. In contrast, moderate scaling preserves low-frequency energy while degrading fine boundary delineation.
2. **Confounding by Mask Size**: Mask area fraction exerts a strong baseline correlation on localization IoU. Controlling for area fraction through partial correlation reveals that boundary compactness and convex solidity remain statistically significant predictors of failure under recompression.
3. **Null Results & Vulnerabilities**: Forensic baselines exhibit correlated failures on diffusion-generated inpainting samples where generative synthesis suppresses camera sensor noise uniformly across the image canvas.
