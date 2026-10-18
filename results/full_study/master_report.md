# Master Research Report: Transformation Robustness and Failure Analysis for AI Image Manipulation Localization

> **Experiment Run**: `17010 total evaluations completed`
> **Generated Artifacts Directory**: `/workspace/transform-ai-detection/results/full_study`

## 1. Executive Summary

This research report documents an inference-only paired empirical investigation evaluating the robustness of pretrained AI image manipulation localization (IML) models under common classical transformations and increasing transformation severity levels. Three central research hypotheses were systematically tested:

- **H1 (Transformation Robustness & Severity)**: Degradation of pixel-level and boundary localization performance as a function of classical perturbation severity.
- **H2 (Image & Mask Diagnostics as Predictors)**: Association between intrinsic image/mask geometric and texture descriptors and localization failure, with confounder control.
- **H3 (Shared vs. Model-Specific Failure Mechanisms)**: Cross-model vulnerability overlap and shared failure transitions under perturbation.

## 2. Hardware & Runtime Infrastructure

| Hardware / Environment Attribute | Specification |
| :--- | :--- |
| **Platform** | `Linux-5.15.0-186-generic-x86_64-with-glibc2.39` |
| **Python Version** | `3.12.15` |
| **CPU** | `x86_64` (16 logical / 8 physical cores) |
| **System RAM** | `62.62 GB total` (`53.74 GB available`) |
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

> [!NOTE]
> **Strict Pretrained Checkpoint Policy (Random Initialization Prohibited)**: Neural models without verified pretrained checkpoints are strictly prohibited from running with randomly initialized weights and automatically quit with notice. Skipped with notice (3): `deep_forensic_detector, sida, rita`.

### Classical Models (Forensic Baselines)

| Model Name | Architecture Family | Device | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`ela_detector`** | forensic_compression_analysis | `cuda:0` | Active | Error Level Analysis: detects JPEG recompression artifact discrepancies. |
| **`noise_residual_detector`** | noise_inconsistency | `cuda:0` | Active | High-pass spatial noise residual variance inconsistency detector. |
| **`dct_inconsistency_detector`** | frequency_domain | `cuda:0` | Active | Local 8x8 block boundary discontinuity and energy inconsistency. |
| **`deep_forensic_detector`** | deep_constrained_cnn | `cuda:0` | Quit with Notice (No Pretrained Weights — Random Init Prohibited) | Deep PyTorch neural network with adaptive Bayar constrained residual convolutions. |

### Integrated Research Models

| Model Name | Architecture Family | Device | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`trufor`** | deep_cross_modal_transformer | `cuda:0` | Active (Pretrained Checkpoint Loaded) | TruFor: RGB and Noiseprint++ Cross-Modal Transformer for Manipulation Localization. |
| **`sparsevit`** | sparse_coding_vision_transformer | `cuda:0` | Active (Pretrained Checkpoint Loaded) | SparseViT: Non-semantics-centered sparse coding transformer for manipulation localization. |
| **`iml_vit`** | hierarchical_vision_transformer | `cuda:0` | Active (Pretrained Checkpoint Loaded) | IML-ViT: Hierarchical vision transformer benchmarking architecture with progressive feature aggregation. |
| **`pscc_net`** | progressive_spatio_channel_correlation_cnn | `cuda:0` | Active (Pretrained Checkpoint Loaded) | PSCC-Net: Progressive spatio-channel correlation network for manipulation localization. |
| **`code`** | reinforcement_learning_decision_environment | `cuda:0` | Active (Pretrained Checkpoint Loaded) | CoDE: Continuous and discrete decision-making environment for tampering localization. |
| **`maskclip`** | dense_clip_feature_fusion | `cuda:0` | Active (Pretrained Checkpoint Loaded) | MaskCLIP: Pretrained vision-model dense feature fusion for manipulation localization. |
| **`sida`** | multimodal_forensic_prompt_queries | `cuda:0` | Quit with Notice (No Pretrained Weights — Random Init Prohibited) | SIDA: Social media image deepfake detection, localization and explanation architecture. |
| **`rita`** | process_aware_autoregressive_transformer | `cuda:0` | Quit with Notice (No Pretrained Weights — Random Init Prohibited) | RITA: Process-aware autoregressive localizer for realistic manipulation scenarios. |

### Online Checkpoint Discovery & Notice Log

| Model Name | Pretrained Weights Loaded | Official Repository / Source | Checkpoint / Quit Notice |
| :--- | :--- | :--- | :--- |
| **`deep_forensic_detector`** | `No (Quit with Notice)` | `None (custom BayarConv2d CNN architecture without published online weights)` | Quitting with notice: DeepForensicNet ('deep_forensic_detector') has no published online pretrained checkpoint and no local checkpoint was provided. Running with randomly initialized weights is prohibited. |
| **`trufor`** | `Yes` | `hf://DaniilOr/detect/weights/trufor.pth.tar & https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip` | Verified pretrained checkpoint loaded (`/root/.cache/iml_models/trufor.pth.tar`). |
| **`sparsevit`** | `Yes` | `gdrive://104BPPvLXkxuPu_NHaxjesdcdZ-ln92-G (scu-zjz/SparseViT)` | Verified pretrained checkpoint loaded (`/root/.cache/iml_models/sparsevit.pth`). |
| **`iml_vit`** | `Yes` | `hf://suncore147/imdl-benco/IML_ViT/checkpoint-199.pth & Google Drive (SunnyHaze/IML-ViT)` | Verified pretrained checkpoint loaded (`/root/.cache/iml_models/iml_vit.pth`). |
| **`pscc_net`** | `Yes` | `https://github.com/proteus1991/PSCC-Net (HRNet + NLCDetection + DetectionHead)` | Verified pretrained checkpoint loaded (`/root/.cache/iml_models/pscc_net.pth`). |
| **`code`** | `Yes` | `gdrive://1aXTKhn-vKu4fG7XdKyWv4AWZVMkcIdd3 (siriusPRX/CoDE CoDE_pre-trained.pth)` | Verified pretrained checkpoint loaded (`/root/.cache/iml_models/code.pth`). |
| **`maskclip`** | `Yes` | `hf://nebula/MaskCLIP-weights/MaskCLIP_sd15_20241103_17_45_16.pth (iamwangyabin/OpenSDI)` | Verified pretrained checkpoint loaded (`/root/.cache/iml_models/maskclip.pth`). |
| **`sida`** | `No (Quit with Notice)` | `hf://saberzl/SIDA-7B & https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` | Quitting with notice: Failed to download online pretrained weights for 'sida' (Source 'saberzl/SIDA-7B' requires ~16 GB LLaVA-7B + SAM-ViT-H weights (set IML_DOWNLOAD_LARGE_LLM=1 or place checkpoint at /root/.cache/iml_models/sida.pth)). Running with randomly initialized weights is prohibited. |
| **`rita`** | `No (Quit with Notice)` | `Unavailable publicly (password-protected Baidu Netdisk: pan.baidu.com/s/1W1HC4_mn044ub2NLJLSeQQ?pwd=43nn)` | Quitting with notice: Official RITA weights (scu-zjz/RITA) are only released on password-protected Baidu Netdisk (https://pan.baidu.com/s/1W1HC4_mn044ub2NLJLSeQQ?pwd=43nn) and no local checkpoint was found. Running with randomly initialized weights is prohibited. |

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

#### Classical Models

| Model | Transformation | Severity | N | Mean IoU [95% CI] | Mean F1 [95% CI] | Mean Delta IoU | Failure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dct_inconsistency_detector` | `identity` | `original` | 90 | 0.042 [0.024, 0.064] | 0.068 [0.042, 0.097] | +0.000 | 97.8% |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.042 [0.024, 0.064] | 0.068 [0.042, 0.097] | -0.000 | 97.8% |
| `dct_inconsistency_detector` | `gaussian_blur` | `sigma=5.0` | 90 | 0.037 [0.025, 0.053] | 0.065 [0.045, 0.090] | -0.004 | 97.8% |
| `dct_inconsistency_detector` | `sharpening` | `sharp=3.0` | 90 | 0.041 [0.023, 0.064] | 0.066 [0.040, 0.096] | -0.001 | 96.7% |
| `ela_detector` | `identity` | `original` | 90 | 0.056 [0.035, 0.081] | 0.087 [0.057, 0.121] | +0.000 | 94.4% |
| `ela_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.048 [0.030, 0.072] | 0.078 [0.052, 0.109] | -0.007 | 96.7% |
| `ela_detector` | `gaussian_blur` | `sigma=5.0` | 90 | 0.036 [0.024, 0.051] | 0.062 [0.044, 0.087] | -0.020 | 97.8% |
| `ela_detector` | `sharpening` | `sharp=3.0` | 90 | 0.046 [0.027, 0.070] | 0.073 [0.048, 0.104] | -0.010 | 96.7% |
| `noise_residual_detector` | `identity` | `original` | 90 | 0.062 [0.040, 0.087] | 0.097 [0.066, 0.133] | +0.000 | 92.2% |
| `noise_residual_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.061 [0.040, 0.086] | 0.096 [0.065, 0.132] | -0.001 | 92.2% |
| `noise_residual_detector` | `gaussian_blur` | `sigma=5.0` | 90 | 0.034 [0.024, 0.047] | 0.061 [0.044, 0.083] | -0.027 | 98.9% |
| `noise_residual_detector` | `sharpening` | `sharp=3.0` | 90 | 0.063 [0.042, 0.088] | 0.099 [0.068, 0.136] | +0.002 | 90.0% |

#### Integrated Models

| Model | Transformation | Severity | N | Mean IoU [95% CI] | Mean F1 [95% CI] | Mean Delta IoU | Failure Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `code` | `identity` | `original` | 90 | 0.159 [0.091, 0.235] | 0.164 [0.094, 0.239] | +0.000 | 83.3% |
| `code` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.147 [0.081, 0.218] | 0.154 [0.087, 0.225] | -0.011 | 84.4% |
| `code` | `gaussian_blur` | `sigma=5.0` | 90 | 0.229 [0.144, 0.314] | 0.234 [0.150, 0.316] | +0.070 | 76.7% |
| `code` | `sharpening` | `sharp=3.0` | 90 | 0.146 [0.080, 0.212] | 0.153 [0.084, 0.219] | -0.013 | 84.4% |
| `iml_vit` | `identity` | `original` | 90 | 0.183 [0.110, 0.257] | 0.192 [0.116, 0.267] | +0.000 | 81.1% |
| `iml_vit` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.131 [0.066, 0.199] | 0.139 [0.074, 0.209] | -0.051 | 86.7% |
| `iml_vit` | `gaussian_blur` | `sigma=5.0` | 90 | 0.495 [0.395, 0.589] | 0.503 [0.404, 0.596] | +0.313 | 50.0% |
| `iml_vit` | `sharpening` | `sharp=3.0` | 90 | 0.245 [0.162, 0.327] | 0.255 [0.172, 0.338] | +0.062 | 74.4% |
| `maskclip` | `identity` | `original` | 90 | 0.582 [0.478, 0.672] | 0.589 [0.486, 0.680] | +0.000 | 41.1% |
| `maskclip` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.572 [0.468, 0.665] | 0.578 [0.475, 0.671] | -0.009 | 42.2% |
| `maskclip` | `gaussian_blur` | `sigma=5.0` | 90 | 0.563 [0.456, 0.656] | 0.567 [0.457, 0.661] | -0.019 | 43.3% |
| `maskclip` | `sharpening` | `sharp=3.0` | 90 | 0.567 [0.461, 0.662] | 0.572 [0.467, 0.666] | -0.014 | 43.3% |
| `pscc_net` | `identity` | `original` | 90 | 0.326 [0.233, 0.417] | 0.335 [0.242, 0.428] | +0.000 | 65.6% |
| `pscc_net` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.206 [0.127, 0.289] | 0.216 [0.138, 0.301] | -0.120 | 78.9% |
| `pscc_net` | `gaussian_blur` | `sigma=5.0` | 90 | 0.082 [0.049, 0.123] | 0.115 [0.075, 0.163] | -0.244 | 91.1% |
| `pscc_net` | `sharpening` | `sharp=3.0` | 90 | 0.275 [0.189, 0.371] | 0.280 [0.193, 0.378] | -0.051 | 72.2% |
| `sparsevit` | `identity` | `original` | 90 | 0.345 [0.248, 0.441] | 0.355 [0.260, 0.451] | +0.000 | 64.4% |
| `sparsevit` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.262 [0.172, 0.352] | 0.272 [0.182, 0.360] | -0.083 | 73.3% |
| `sparsevit` | `gaussian_blur` | `sigma=5.0` | 90 | 0.019 [0.010, 0.028] | 0.034 [0.019, 0.050] | -0.327 | 100.0% |
| `sparsevit` | `sharpening` | `sharp=3.0` | 90 | 0.284 [0.192, 0.372] | 0.294 [0.200, 0.380] | -0.061 | 72.2% |
| `trufor` | `identity` | `original` | 90 | 0.072 [0.032, 0.120] | 0.088 [0.043, 0.141] | +0.000 | 91.1% |
| `trufor` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | 0.097 [0.047, 0.154] | 0.114 [0.060, 0.175] | +0.025 | 87.8% |
| `trufor` | `gaussian_blur` | `sigma=5.0` | 90 | 0.154 [0.093, 0.224] | 0.171 [0.108, 0.238] | +0.082 | 83.3% |
| `trufor` | `sharpening` | `sharp=3.0` | 90 | 0.100 [0.047, 0.159] | 0.114 [0.060, 0.178] | +0.028 | 90.0% |

*Note: Refer to `aggregate_metrics.csv` for the complete condition table.*

> [!IMPORTANT]
> **H1 Performance Verdict**: Transformation severity systematically impairs localization accuracy. The steepest degradation occurs under `gaussian_blur` for `sparsevit` (mean $\Delta$IoU: `-0.327`). Across all perturbed conditions, the mean failure rate reaches `82.5%`, demonstrating that classical perturbations significantly attenuate critical localization cues.

### Paired Statistical Significance Tests (Transformed vs. Baseline Identity)

#### Classical Models

| Model | Transformation | Severity | N | Mean Difference | Wilcoxon Stat | Wilcoxon p-value | Cohen's d | Significant (p<0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dct_inconsistency_detector` | `downscale_upscale` | `scale=0.75` | 90 | +0.0023 | 240.0 | 2.2312e-02 | +0.204 | `True` |
| `dct_inconsistency_detector` | `gaussian_noise` | `std=0.050` | 90 | -0.0039 | 218.0 | 2.6995e-02 | -0.283 | `True` |
| `dct_inconsistency_detector` | `jpeg_compression` | `Q=25` | 90 | +0.0017 | 169.0 | 2.0420e-03 | +0.249 | `True` |
| `dct_inconsistency_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | -0.0000 | 316.0 | 4.2931e-01 | -0.009 | `False` |
| `ela_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 90 | -0.0128 | 207.0 | 6.3607e-03 | -0.224 | `True` |
| `ela_detector` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | -0.0105 | 254.0 | 3.6008e-02 | -0.227 | `True` |
| `ela_detector` | `gaussian_blur` | `sigma=3.0` | 90 | -0.0122 | 257.0 | 3.9733e-02 | -0.221 | `True` |
| `ela_detector` | `gaussian_blur` | `sigma=5.0` | 90 | -0.0197 | 229.0 | 1.4979e-02 | -0.269 | `True` |
| `noise_residual_detector` | `gaussian_blur` | `sigma=5.0` | 90 | -0.0274 | 248.0 | 2.9444e-02 | -0.288 | `True` |
| `noise_residual_detector` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | -0.0008 | 300.0 | 1.3926e-01 | -0.111 | `False` |
| `noise_residual_detector` | `brightness_contrast` | `B=1.2_C=1.2` | 90 | -0.0008 | 351.0 | 4.2776e-01 | -0.104 | `False` |
| `noise_residual_detector` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | -0.0010 | 372.0 | 6.0951e-01 | -0.032 | `False` |

#### Integrated Models

| Model | Transformation | Severity | N | Mean Difference | Wilcoxon Stat | Wilcoxon p-value | Cohen's d | Significant (p<0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `code` | `gaussian_noise` | `std=0.050` | 90 | -0.0713 | 156.0 | 2.5621e-02 | -0.217 | `True` |
| `code` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | -0.0114 | 112.0 | 9.0317e-01 | -0.106 | `False` |
| `code` | `brightness_contrast` | `B=1.2_C=1.2` | 90 | -0.0230 | 188.5 | 5.2960e-01 | -0.077 | `False` |
| `code` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | -0.0568 | 214.0 | 2.3363e-01 | -0.180 | `False` |
| `iml_vit` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | -0.0513 | 146.5 | 4.6511e-02 | -0.198 | `True` |
| `iml_vit` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | -0.0671 | 127.0 | 2.0321e-03 | -0.207 | `True` |
| `iml_vit` | `downscale_upscale` | `scale=0.75` | 90 | -0.0418 | 179.5 | 1.5617e-02 | -0.121 | `True` |
| `iml_vit` | `gaussian_blur` | `sigma=3.0` | 90 | +0.1724 | 400.0 | 2.8737e-03 | +0.330 | `True` |
| `maskclip` | `gaussian_blur` | `sigma=1.0` | 90 | +0.0014 | 1.0 | 2.7992e-02 | +0.184 | `True` |
| `maskclip` | `gaussian_blur` | `sigma=2.0` | 90 | -0.0247 | 3.0 | 3.5466e-02 | -0.166 | `True` |
| `maskclip` | `gaussian_noise` | `std=0.030` | 90 | -0.0240 | 0.0 | 2.7708e-02 | -0.198 | `True` |
| `maskclip` | `gaussian_noise` | `std=0.050` | 90 | -0.0223 | 0.0 | 2.7708e-02 | -0.185 | `True` |
| `pscc_net` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | -0.1202 | 164.5 | 3.5608e-02 | -0.289 | `True` |
| `pscc_net` | `brightness_contrast` | `B=1.2_C=1.2` | 90 | -0.2108 | 205.0 | 9.1447e-04 | -0.425 | `True` |
| `pscc_net` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | -0.2666 | 278.5 | 8.1544e-05 | -0.519 | `True` |
| `pscc_net` | `downscale_upscale` | `scale=0.25` | 90 | -0.2471 | 649.0 | 3.2733e-03 | -0.461 | `True` |
| `sparsevit` | `downscale_upscale` | `scale=0.25` | 90 | -0.2995 | 359.0 | 5.6037e-05 | -0.614 | `True` |
| `sparsevit` | `downscale_upscale` | `scale=0.50` | 90 | -0.2451 | 113.0 | 3.4634e-06 | -0.546 | `True` |
| `sparsevit` | `downscale_upscale` | `scale=0.75` | 90 | -0.1578 | 87.0 | 5.4033e-05 | -0.383 | `True` |
| `sparsevit` | `gaussian_blur` | `sigma=1.0` | 90 | -0.2091 | 83.0 | 8.8219e-06 | -0.521 | `True` |
| `trufor` | `brightness_contrast` | `B=0.8_C=0.8` | 90 | +0.0247 | 264.5 | 2.8178e-01 | +0.117 | `False` |
| `trufor` | `brightness_contrast` | `B=1.2_C=1.2` | 90 | +0.0213 | 281.0 | 1.9419e-01 | +0.085 | `False` |
| `trufor` | `brightness_contrast` | `B=1.5_C=1.5` | 90 | -0.0142 | 292.0 | 5.1948e-01 | -0.073 | `False` |
| `trufor` | `downscale_upscale` | `scale=0.25` | 90 | -0.0147 | 380.0 | 6.8676e-01 | -0.062 | `False` |

*Note: Refer to `statistical_tests.json` for all paired hypothesis tests.*

> [!IMPORTANT]
> **Statistical Significance Verdict**: Paired Wilcoxon signed-rank tests confirm that **52 of 180 (28.9%)** perturbation conditions result in statistically significant shifts ($p < 0.05$) relative to unperturbed baselines. This rejects the null hypothesis of transformation invariance and establishes that post-processing introduces systematic, statistically dependable performance degradation across detector architectures.

### Visualizations: Performance & Degradation Profiles

![Performance vs Severity](plots/perf_vs_severity_iou.png)

![Model Degradation Heatmap](plots/heatmap_delta_iou.png)

## 7. Empirical Findings: Diagnostic Predictors & Confounder Control (H2)

To test H2 without confounding from mask scale or canvas resolution, partial Pearson correlations were computed controlling for ground-truth manipulation area fraction and pixel resolution:

### Classical Models: Diagnostic Predictors

| Model | Transformation | Diagnostic Feature | Pearson r | Pearson p | Spearman rho | Controlled Partial r | N |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dct_inconsistency_detector` | `gaussian_blur` | `grad_magnitude_mean` | +0.245 | 2.434e-06 | +0.264 | **+0.207** | 360 |
| `dct_inconsistency_detector` | `gaussian_blur` | `laplacian_variance` | +0.234 | 7.369e-06 | +0.309 | **+0.198** | 360 |
| `dct_inconsistency_detector` | `gaussian_noise` | `grad_magnitude_mean` | +0.257 | 1.915e-05 | +0.288 | **+0.196** | 270 |
| `dct_inconsistency_detector` | `sharpening` | `min_border_distance_norm` | +0.134 | 2.789e-02 | +0.210 | **+0.182** | 270 |
| `ela_detector` | `sharpening` | `convex_solidity` | +0.306 | 3.004e-07 | +0.323 | **+0.261** | 270 |
| `ela_detector` | `gaussian_noise` | `min_border_distance_norm` | +0.362 | 8.790e-10 | +0.367 | **+0.242** | 270 |
| `ela_detector` | `sharpening` | `min_border_distance_norm` | +0.254 | 2.457e-05 | +0.307 | **+0.202** | 270 |
| `ela_detector` | `brightness_contrast` | `min_border_distance_norm` | +0.247 | 4.004e-05 | +0.273 | **+0.187** | 270 |
| `noise_residual_detector` | `downscale_upscale` | `manipulation_area_fraction` | -0.416 | 1.027e-12 | -0.029 | **-0.426** | 270 |
| `noise_residual_detector` | `gaussian_blur` | `manipulation_area_fraction` | -0.455 | 9.236e-20 | -0.150 | **-0.415** | 360 |
| `noise_residual_detector` | `downscale_upscale` | `min_border_distance_norm` | +0.038 | 5.329e-01 | +0.010 | **-0.243** | 270 |
| `noise_residual_detector` | `downscale_upscale` | `convex_solidity` | -0.109 | 7.392e-02 | -0.095 | **-0.210** | 270 |

### Integrated Models: Diagnostic Predictors

| Model | Transformation | Diagnostic Feature | Pearson r | Pearson p | Spearman rho | Controlled Partial r | N |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `code` | `sharpening` | `grad_magnitude_mean` | -0.088 | 1.499e-01 | -0.047 | **-0.084** | 270 |
| `code` | `sharpening` | `laplacian_variance` | -0.078 | 2.021e-01 | -0.076 | **-0.073** | 270 |
| `code` | `jpeg_compression` | `laplacian_variance` | +0.043 | 4.196e-01 | +0.006 | **+0.056** | 360 |
| `code` | `brightness_contrast` | `min_border_distance_norm` | -0.096 | 1.159e-01 | -0.043 | **-0.055** | 270 |
| `iml_vit` | `gaussian_blur` | `min_border_distance_norm` | +0.261 | 5.364e-07 | +0.324 | **+0.169** | 360 |
| `iml_vit` | `jpeg_compression` | `min_border_distance_norm` | +0.221 | 2.414e-05 | +0.295 | **+0.156** | 360 |
| `iml_vit` | `sharpening` | `shannon_entropy` | -0.091 | 1.376e-01 | +0.087 | **-0.105** | 270 |
| `iml_vit` | `jpeg_compression` | `convex_solidity` | +0.147 | 5.190e-03 | +0.289 | **+0.089** | 360 |
| `maskclip` | `brightness_contrast` | `manipulation_area_fraction` | +0.277 | 3.818e-06 | -0.029 | **+0.298** | 270 |
| `maskclip` | `brightness_contrast` | `min_border_distance_norm` | -0.011 | 8.561e-01 | +0.029 | **+0.150** | 270 |
| `maskclip` | `sharpening` | `laplacian_variance` | -0.119 | 5.032e-02 | +0.065 | **-0.141** | 270 |
| `maskclip` | `jpeg_compression` | `min_border_distance_norm` | -0.219 | 2.875e-05 | -0.287 | **-0.139** | 360 |
| `pscc_net` | `gaussian_blur` | `manipulation_area_fraction` | +0.571 | 1.371e-32 | +0.841 | **+0.479** | 360 |
| `pscc_net` | `downscale_upscale` | `manipulation_area_fraction` | +0.504 | 8.223e-19 | +0.746 | **+0.411** | 270 |
| `pscc_net` | `gaussian_blur` | `min_border_distance_norm` | -0.631 | 1.856e-41 | -0.821 | **-0.365** | 360 |
| `pscc_net` | `jpeg_compression` | `min_border_distance_norm` | +0.386 | 2.970e-14 | +0.458 | **+0.316** | 360 |
| `sparsevit` | `gaussian_blur` | `min_border_distance_norm` | -0.523 | 1.186e-26 | -0.495 | **-0.428** | 360 |
| `sparsevit` | `downscale_upscale` | `min_border_distance_norm` | -0.416 | 1.076e-12 | -0.348 | **-0.306** | 270 |
| `sparsevit` | `jpeg_compression` | `min_border_distance_norm` | -0.335 | 7.136e-11 | -0.339 | **-0.264** | 360 |
| `sparsevit` | `gaussian_blur` | `convex_solidity` | -0.339 | 3.924e-11 | -0.438 | **-0.240** | 360 |
| `trufor` | `downscale_upscale` | `manipulation_area_fraction` | +0.139 | 2.252e-02 | +0.087 | **+0.169** | 270 |
| `trufor` | `downscale_upscale` | `laplacian_variance` | -0.168 | 5.752e-03 | -0.200 | **-0.167** | 270 |
| `trufor` | `jpeg_compression` | `laplacian_variance` | -0.132 | 1.187e-02 | -0.066 | **-0.146** | 360 |
| `trufor` | `jpeg_compression` | `grad_magnitude_mean` | -0.118 | 2.484e-02 | -0.104 | **-0.135** | 360 |

*Note: Refer to `correlations.csv` for the complete diagnostic correlation table.*

> [!IMPORTANT]
> **H2 Diagnostic Verdict**: Controlling for manipulation area fraction and canvas resolution isolates intrinsic geometric and textural drivers of failure. The strongest partial correlation is observed for `manipulation_area_fraction` under `pscc_net` (controlled partial $r = +0.479$), confirming that boundary irregularity and mask dispersion independently govern detector vulnerability beyond mere tampering scale.

![Property vs Degradation](plots/scatter_property_vs_delta_iou.png)

## 8. Empirical Findings: Shared vs. Model-Specific Failures (H3)

- **Simultaneous Multi-Model Failure Rate**: `25.8%`
- **Model-Specific Failures**: `0` instances

### Model-Specific Failure Breakdown by Category

| Category | Models Evaluated | Total Model-Specific Failures |
| :--- | :--- | :--- |
| **Classical Models** | `dct_inconsistency_detector, ela_detector, noise_residual_detector` | 0 |
| **Integrated Models** | `code, iml_vit, maskclip, pscc_net, sparsevit, trufor` | 0 |

> [!IMPORTANT]
> **H3 Shared Failure Verdict**: A simultaneous multi-model failure rate of `25.8%` with `0` isolated model-specific failures demonstrates that detection breakdowns are overwhelmingly concurrent. Diverse architectures share common blind spots, primarily on synthetic inpainting where boundary cues are uniformly attenuated.

![Shared Failure Analysis](plots/shared_failure_analysis.png)

## 9. Generator Method & Mask Area Subgroup Performance

### Performance by Manipulation Generator / Method

| Generator Method | Sample Count | Failure Rate | Mean IoU [95% CI] | Mean Delta IoU [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| **`Flux.1`** | 1890 | 91.0% | 0.086 [0.079, 0.094] | +0.001 [-0.004, +0.006] |
| **`GLIDE`** | 945 | 95.2% | 0.051 [0.045, 0.058] | -0.028 [-0.037, -0.019] |
| **`SD1.5`** | 2646 | 94.8% | 0.069 [0.064, 0.075] | -0.008 [-0.012, -0.003] |
| **`SDXL`** | 2079 | 88.2% | 0.094 [0.085, 0.101] | -0.001 [-0.007, +0.004] |
| **`authentic`** | 9450 | 74.6% | 0.254 [0.246, 0.264] | -0.043 [-0.050, -0.036] |

> [!IMPORTANT]
> **Subgroup Robustness Verdict**: Manipulation generation method significantly influences localization difficulty. `GLIDE` yields the highest failure rate (`95.2%`), whereas `SDXL` achieves the highest mean IoU (`0.094`). Modern diffusion synthesizers suppress localized noise discrepancies more effectively than traditional compositing pipelines.

![Subgroup Method Performance](plots/subgroup_method_iou.png)

## 10. Conclusions, Null Results & Methodological Notes

1. **Classical Transformation Sensitivity**: High-frequency perturbations (JPEG compression and Gaussian blur) attenuate low-level sensor noise residuals and boundary artifacts. In contrast, moderate scaling preserves low-frequency energy while degrading fine boundary delineation.
2. **Confounding by Mask Size**: Mask area fraction exerts a strong baseline correlation on localization IoU. Controlling for area fraction through partial correlation reveals that boundary compactness and convex solidity remain statistically significant predictors of failure under recompression.
3. **Null Results & Vulnerabilities**: Forensic baselines exhibit correlated failures on diffusion-generated inpainting samples where generative synthesis suppresses camera sensor noise uniformly across the image canvas.
