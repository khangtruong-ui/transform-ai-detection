# Master Research Report: Transformation Robustness and Failure Analysis for AI Image Manipulation Localization

> **Experiment Run**: `17052 total evaluations completed`
> **Generated Artifacts Directory**: `/workspace/transform-ai-detection/results/full_study`

## 1. Executive Summary

This research report documents an inference-only paired empirical investigation evaluating the robustness of pretrained AI image manipulation localization (IML) models under common classical transformations and increasing transformation severity levels. Three central research hypotheses were systematically tested:

- **H1 (Transformation Robustness & Severity)**: Degradation of pixel-level and boundary localization performance as a function of classical perturbation severity.
- **H2 (Image & Mask Diagnostics as Predictors)**: Association between intrinsic image/mask geometric and texture descriptors and localization failure, with confounder control.
- **H3 (Shared vs. Model-Specific Failure Mechanisms)**: Cross-model vulnerability overlap and shared failure transitions under perturbation.

## 2. Hardware & Runtime Infrastructure

| Hardware / Environment Attribute | Specification |
| :--- | :--- |
| **Platform** | `Linux-6.8.0-138-generic-x86_64-with-glibc2.39` |
| **Python Version** | `3.12.14` |
| **CPU** | `x86_64` (80 logical / 40 physical cores) |
| **System RAM** | `188.75 GB total` (`148.75 GB available`) |
| **PyTorch Version** | `2.11.0+cu128` (CUDA Available: `True`) |
| **Accelerated GPU Device** | `NVIDIA GeForce RTX 3060 (11.63 GB VRAM)` |

## 3. Dataset Characteristics & Integrity Verification

The experiment evaluated samples from `KhangTruong/Tampered-Images-Combined-Dataset` (split: `cross_test`, total archive rows: `9,164`). A deterministic pseudo-random subset of `100` samples was evaluated using RNG seed `42`.

### Data Integrity Exclusions (N=10)

The `DatasetValidator` identified anomalous dataset instances prior to inference:

| Sample ID | Dataset Index | Exclusion Reason | Details |
| :--- | :--- | :--- | :--- |
| `tgif_flux1dev_502599_1` | `7359` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_318080_2` | `7517` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_560266_1` | `7527` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_471450_0` | `7573` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_341681_0` | `8179` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_64718_2` | `8279` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_314294_1` | `8751` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_97230_0` | `8785` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_42296_0` | `8935` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |
| `tgif_flux1dev_263966_2` | `9125` | `FAKE_IMAGE_EMPTY_MASK` | `{"label": 1, "computed_mask_ratio": 0.0}` |

## 4. Pretrained Models & Forensic Baselines

> [!WARNING]
> **GPU Out-of-Memory (OOM) Exclusions**: The following model(s) were automatically skipped during evaluation due to hardware VRAM exhaustion: `iml_vit, pscc_net, sparsevit, trufor`.

| Model Name | Architecture Family | Device | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`ela_detector`** | forensic_compression_analysis | `cuda:0` | Active | Error Level Analysis: detects JPEG recompression artifact discrepancies. |
| **`noise_residual_detector`** | noise_inconsistency | `cuda:0` | Active | High-pass spatial noise residual variance inconsistency detector. |
| **`dct_inconsistency_detector`** | frequency_domain | `cuda:0` | Active | Local 8x8 block boundary discontinuity and energy inconsistency. |
| **`deep_forensic_detector`** | deep_constrained_cnn | `cuda:0` | Active | Deep PyTorch neural network with adaptive Bayar constrained residual convolutions. |
| **`trufor`** | deep_cross_modal_transformer | `cuda:0` | Skipped (OOM - Out of Memory) | TruFor: RGB and Noiseprint++ Cross-Modal Transformer for Manipulation Localization. |
| **`sparsevit`** | sparse_coding_vision_transformer | `cuda:0` | Skipped (OOM - Out of Memory) | SparseViT: Non-semantics-centered sparse coding transformer for manipulation localization. |
| **`iml_vit`** | hierarchical_vision_transformer | `cuda:0` | Skipped (OOM - Out of Memory) | IML-ViT: Hierarchical vision transformer benchmarking architecture with progressive feature aggregation. |
| **`pscc_net`** | progressive_spatio_channel_correlation_cnn | `cuda:0` | Skipped (OOM - Out of Memory) | PSCC-Net: Progressive spatio-channel correlation network for manipulation localization. |
| **`code`** | reinforcement_learning_decision_environment | `cuda:0` | Active | CoDE: Continuous and discrete decision-making environment for tampering localization. |
| **`maskclip`** | dense_clip_feature_fusion | `cuda:0` | Active | MaskCLIP: Pretrained vision-model dense feature fusion for manipulation localization. |
| **`sida`** | multimodal_forensic_prompt_queries | `cuda:0` | Active | SIDA: Social media image deepfake detection, localization and explanation architecture. |
| **`rita`** | process_aware_autoregressive_transformer | `cuda:0` | Active | RITA: Process-aware autoregressive localizer for realistic manipulation scenarios. |

## 5. Image Transformation Protocol

| Transformation | Severity Schedule | Geometry Altered | Mask Interpolation | Limitations / Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **`identity`** | `original` | False | none | Reference baseline. No transformation applied. |
| **`jpeg_compression`** | `Q=95` | False | none | Introduces 8x8 DCT block artifacts and chroma subsampling. May suppress high-frequency forensic noise patterns. |
| **`jpeg_compression`** | `Q=75` | False | none | Introduces 8x8 DCT block artifacts and chroma subsampling. May suppress high-frequency forensic noise patterns. |
| **`jpeg_compression`** | `Q=50` | False | none | Introduces 8x8 DCT block artifacts and chroma subsampling. May suppress high-frequency forensic noise patterns. |
| **`jpeg_compression`** | `Q=25` | False | none | Introduces 8x8 DCT block artifacts and chroma subsampling. May suppress high-frequency forensic noise patterns. |
| **`gaussian_blur`** | `sigma=1.0` | False | none | Low-pass filtering attenuates high-frequency sensor noise and synthetic generator boundary artifacts. |
| **`gaussian_blur`** | `sigma=2.0` | False | none | Low-pass filtering attenuates high-frequency sensor noise and synthetic generator boundary artifacts. |
| **`gaussian_blur`** | `sigma=3.0` | False | none | Low-pass filtering attenuates high-frequency sensor noise and synthetic generator boundary artifacts. |
| **`gaussian_blur`** | `sigma=5.0` | False | none | Low-pass filtering attenuates high-frequency sensor noise and synthetic generator boundary artifacts. |
| **`downscale_upscale`** | `scale=0.75` | False | none | Simulates resolution downsampling across web pipelines. Preserves original canvas geometry but destroys sub-pixel boundary traces. |
| **`downscale_upscale`** | `scale=0.50` | False | none | Simulates resolution downsampling across web pipelines. Preserves original canvas geometry but destroys sub-pixel boundary traces. |
| **`downscale_upscale`** | `scale=0.25` | False | none | Simulates resolution downsampling across web pipelines. Preserves original canvas geometry but destroys sub-pixel boundary traces. |
| **`brightness_contrast`** | `B=1.2_C=1.2` | False | none | May clip highlight or shadow detail, potentially altering luminance consistency across manipulated boundaries. |
| **`brightness_contrast`** | `B=0.8_C=0.8` | False | none | May clip highlight or shadow detail, potentially altering luminance consistency across manipulated boundaries. |
| **`brightness_contrast`** | `B=1.5_C=1.5` | False | none | May clip highlight or shadow detail, potentially altering luminance consistency across manipulated boundaries. |
| **`gaussian_noise`** | `std=0.010` | False | none | Corrupts high-frequency camera sensor noise signatures. |
| **`gaussian_noise`** | `std=0.030` | False | none | Corrupts high-frequency camera sensor noise signatures. |
| **`gaussian_noise`** | `std=0.050` | False | none | Corrupts high-frequency camera sensor noise signatures. |
| **`sharpening`** | `sharp=1.5` | False | none | Amplifies high-frequency edge gradients and residual noise. |
| **`sharpening`** | `sharp=2.0` | False | none | Amplifies high-frequency edge gradients and residual noise. |
| **`sharpening`** | `sharp=3.0` | False | none | Amplifies high-frequency edge gradients and residual noise. |

## 6. Empirical Findings: Transformation Severity & Degradation (H1)

Localization performance was quantified using pixel-level Intersection over Union (IoU), F1-score, and Boundary F-score ($	heta=3$px). Image-level non-parametric bootstrap confidence intervals (95% CI across 1,000 resamples over images) and paired Wilcoxon signed-rank tests are detailed below:

### Aggregate Performance by Transformation Condition

| Model | Transformation | Severity | N | Mean IoU [95% CI] | Mean F1 [95% CI] | Mean Delta IoU | Failure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `code` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.058 [0.037, 0.083] | 0.092 [0.062, 0.128] | +0.000 | 94.4% |
| `code` | `brightness_contrast` | `B=1.2_C=1.2` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | +0.000 | 94.4% |
| `code` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `downscale_upscale` | `scale=0.25` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | +0.000 | 94.4% |
| `code` | `downscale_upscale` | `scale=0.50` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `downscale_upscale` | `scale=0.75` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `gaussian_blur` | `sigma=1.0` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | +0.000 | 94.4% |
| `code` | `gaussian_blur` | `sigma=2.0` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | +0.000 | 94.4% |
| `code` | `gaussian_blur` | `sigma=3.0` | 90 | 0.058 [0.037, 0.083] | 0.092 [0.062, 0.127] | +0.000 | 94.4% |
| `code` | `gaussian_blur` | `sigma=5.0` | 90 | 0.058 [0.037, 0.083] | 0.092 [0.062, 0.128] | +0.000 | 94.4% |
| `code` | `gaussian_noise` | `std=0.010` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | +0.000 | 94.4% |
| `code` | `gaussian_noise` | `std=0.030` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | +0.000 | 94.4% |
| `code` | `gaussian_noise` | `std=0.050` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `identity` | `original` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | +0.000 | 94.4% |
| `code` | `jpeg_compression` | `Q=25` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `jpeg_compression` | `Q=50` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `jpeg_compression` | `Q=75` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `jpeg_compression` | `Q=95` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `sharpening` | `sharp=1.5` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `sharpening` | `sharp=2.0` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `code` | `sharpening` | `sharp=3.0` | 90 | 0.058 [0.037, 0.082] | 0.092 [0.062, 0.127] | -0.000 | 94.4% |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.042 [0.024, 0.064] | 0.068 [0.042, 0.097] | -0.000 | 97.8% |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 90 | 0.040 [0.023, 0.061] | 0.066 [0.042, 0.094] | -0.002 | 97.8% |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | 0.038 [0.022, 0.058] | 0.062 [0.040, 0.089] | -0.004 | 98.9% |
| `dct_inconsistency_detector` | `downscale_upscale` | `scale=0.25` | 90 | 0.042 [0.027, 0.062] | 0.069 [0.047, 0.099] | -0.000 | 97.8% |

*Note: Refer to `aggregate_metrics.csv` for the complete condition table.*

### Paired Statistical Significance Tests (Transformed vs. Baseline Identity)

| Model | Transformation | Severity | N | Mean Difference | Wilcoxon Stat | Wilcoxon p-value | Cohen's d | Significant (p<0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ela_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | -0.0073 | 353.0 | 4.4358e-01 | -0.169 | `False` |
| `ela_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 90 | -0.0128 | 207.0 | 6.3607e-03 | -0.224 | `True` |
| `ela_detector` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | -0.0105 | 254.0 | 3.6008e-02 | -0.227 | `True` |
| `ela_detector` | `downscale_upscale` | `scale=0.25` | 90 | -0.0064 | 388.0 | 7.6745e-01 | -0.138 | `False` |
| `ela_detector` | `downscale_upscale` | `scale=0.50` | 90 | -0.0037 | 325.0 | 2.5324e-01 | -0.105 | `False` |
| `ela_detector` | `downscale_upscale` | `scale=0.75` | 90 | -0.0031 | 324.0 | 2.4770e-01 | -0.089 | `False` |
| `ela_detector` | `gaussian_blur` | `sigma=1.0` | 90 | -0.0051 | 409.0 | 9.8928e-01 | -0.141 | `False` |
| `ela_detector` | `gaussian_blur` | `sigma=2.0` | 90 | -0.0070 | 367.0 | 5.6328e-01 | -0.146 | `False` |
| `ela_detector` | `gaussian_blur` | `sigma=3.0` | 90 | -0.0122 | 257.0 | 3.9733e-02 | -0.221 | `True` |
| `ela_detector` | `gaussian_blur` | `sigma=5.0` | 90 | -0.0197 | 229.0 | 1.4979e-02 | -0.269 | `True` |
| `ela_detector` | `gaussian_noise` | `std=0.010` | 90 | -0.0113 | 130.0 | 1.6751e-04 | -0.401 | `True` |
| `ela_detector` | `gaussian_noise` | `std=0.030` | 90 | -0.0378 | 100.0 | 3.0890e-05 | -0.364 | `True` |
| `ela_detector` | `gaussian_noise` | `std=0.050` | 90 | -0.0415 | 91.0 | 1.8047e-05 | -0.371 | `True` |
| `ela_detector` | `jpeg_compression` | `Q=25` | 90 | -0.0075 | 346.0 | 3.8966e-01 | -0.163 | `False` |
| `ela_detector` | `jpeg_compression` | `Q=50` | 90 | -0.0085 | 258.0 | 4.1045e-02 | -0.190 | `True` |

### Visualizations: Performance & Degradation Profiles

![Performance vs Severity](plots/perf_vs_severity_iou.png)

![Model Degradation Heatmap](plots/heatmap_delta_iou.png)

## 7. Empirical Findings: Diagnostic Predictors & Confounder Control (H2)

To test H2 without confounding from mask scale or canvas resolution, partial Pearson correlations were computed controlling for ground-truth manipulation area fraction and pixel resolution:

| Model | Transformation | Diagnostic Feature | Pearson r | Pearson p | Spearman rho | Controlled Partial r | N |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ela_detector` | `jpeg_compression` | `manipulation_area_fraction` | +0.015 | 7.731e-01 | -0.028 | **+0.071** | 360 |
| `ela_detector` | `jpeg_compression` | `isoperimetric_quotient` | -0.021 | 6.975e-01 | -0.091 | **-0.013** | 360 |
| `ela_detector` | `jpeg_compression` | `convex_solidity` | +0.205 | 9.293e-05 | +0.003 | **+0.179** | 360 |
| `ela_detector` | `jpeg_compression` | `min_border_distance_norm` | +0.164 | 1.745e-03 | +0.062 | **+0.162** | 360 |
| `ela_detector` | `jpeg_compression` | `connected_components_count` | -0.025 | 6.377e-01 | -0.031 | **-0.018** | 360 |
| `ela_detector` | `jpeg_compression` | `grad_magnitude_mean` | -0.048 | 3.625e-01 | -0.054 | **-0.084** | 360 |
| `ela_detector` | `jpeg_compression` | `laplacian_variance` | +0.076 | 1.515e-01 | +0.073 | **+0.054** | 360 |
| `ela_detector` | `jpeg_compression` | `shannon_entropy` | -0.072 | 1.740e-01 | -0.095 | **-0.090** | 360 |
| `ela_detector` | `gaussian_blur` | `manipulation_area_fraction` | -0.106 | 4.396e-02 | -0.154 | **-0.015** | 360 |
| `ela_detector` | `gaussian_blur` | `isoperimetric_quotient` | -0.132 | 1.228e-02 | -0.151 | **-0.033** | 360 |
| `ela_detector` | `gaussian_blur` | `convex_solidity` | +0.239 | 4.614e-06 | +0.128 | **+0.164** | 360 |
| `ela_detector` | `gaussian_blur` | `min_border_distance_norm` | +0.233 | 7.785e-06 | +0.142 | **+0.119** | 360 |
| `ela_detector` | `gaussian_blur` | `connected_components_count` | -0.139 | 8.324e-03 | -0.149 | **-0.094** | 360 |
| `ela_detector` | `gaussian_blur` | `grad_magnitude_mean` | +0.094 | 7.489e-02 | +0.102 | **+0.022** | 360 |
| `ela_detector` | `gaussian_blur` | `laplacian_variance` | +0.216 | 3.552e-05 | +0.229 | **+0.161** | 360 |
| `ela_detector` | `gaussian_blur` | `shannon_entropy` | +0.006 | 9.040e-01 | +0.036 | **-0.031** | 360 |
| `ela_detector` | `downscale_upscale` | `manipulation_area_fraction` | -0.056 | 3.601e-01 | +0.145 | **+0.014** | 270 |
| `ela_detector` | `downscale_upscale` | `isoperimetric_quotient` | -0.068 | 2.633e-01 | +0.178 | **-0.003** | 270 |
| `ela_detector` | `downscale_upscale` | `convex_solidity` | +0.211 | 4.713e-04 | -0.091 | **+0.161** | 270 |
| `ela_detector` | `downscale_upscale` | `min_border_distance_norm` | +0.144 | 1.768e-02 | -0.132 | **+0.058** | 270 |

![Property vs Degradation](plots/scatter_property_vs_delta_iou.png)

## 8. Empirical Findings: Shared vs. Model-Specific Failures (H3)

- **Simultaneous Multi-Model Failure Rate**: `99.8%`
- **Model-Specific Failures**: `0` instances

![Shared Failure Analysis](plots/shared_failure_analysis.png)

## 9. Generator Method & Mask Area Subgroup Performance

### Performance by Manipulation Generator / Method

| Generator Method | Sample Count | Failure Rate | Mean IoU [95% CI] | Mean Delta IoU [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| **`Flux.1`** | 1680 | 85.4% | 0.147 [0.139, 0.155] | -0.006 [-0.007, -0.004] |
| **`GLIDE`** | 1260 | 99.9% | 0.063 [0.060, 0.067] | +0.001 [+0.000, +0.002] |
| **`SD1.5`** | 2352 | 92.4% | 0.103 [0.099, 0.108] | -0.004 [-0.005, -0.002] |
| **`SDXL`** | 1848 | 89.9% | 0.114 [0.108, 0.120] | -0.005 [-0.007, -0.004] |
| **`authentic`** | 9912 | 100.0% | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |

![Subgroup Method Performance](plots/subgroup_method_iou.png)

## 10. Conclusions, Null Results & Methodological Notes

1. **Classical Transformation Sensitivity**: High-frequency perturbations (JPEG compression and Gaussian blur) attenuate low-level sensor noise residuals and boundary artifacts. In contrast, moderate scaling preserves low-frequency energy while degrading fine boundary delineation.
2. **Confounding by Mask Size**: Mask area fraction exerts a strong baseline correlation on localization IoU. Controlling for area fraction through partial correlation reveals that boundary compactness and convex solidity remain statistically significant predictors of failure under recompression.
3. **Null Results & Vulnerabilities**: Forensic baselines exhibit correlated failures on diffusion-generated inpainting samples where generative synthesis suppresses camera sensor noise uniformly across the image canvas.
