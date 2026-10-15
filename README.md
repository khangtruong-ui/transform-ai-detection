# Transformation Robustness and Failure Analysis for AI Image Manipulation Localization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-47%20passing-brightgreen.svg)](tests/)

A reproducible, inference-only research codebase designed to study how classical image transformations impact pretrained AI image manipulation localization (IML) models, and how intrinsic image/mask geometric and morphological properties govern localization success and failure.

---

## 1. Research Objectives & Hypotheses

The framework evaluates three primary research hypotheses without assuming their validity:

- **H1 (Transformation Robustness & Severity)**: How does each pretrained model's localization performance degrade under classical transformations (JPEG recompression, Gaussian blur, downscale-upscale, lighting changes, noise, sharpening, flips, and crops) across increasing severity levels?
- **H2 (Image & Mask Characteristics as Failure Predictors)**: Are pre-inference image properties (manipulation area fraction, boundary compactness/isoperimetric quotient, connected-component count, distance to canvas borders, and gradient texture statistics) significantly correlated with localization degradation?
- **H3 (Shared vs. Model-Specific Failure Mechanisms)**: Do transformations induce correlated failures across distinct model architectures, or are failure modes predominantly model-specific?

---

## 2. System Architecture

```
                                      ┌──────────────────────────────────────────────┐
                                      │              Dataset Adapter                 │
                                      │ (parquet-dataset-loader / Synthetic / Local) │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                                             ▼
                                      ┌──────────────────────────────────────────────┐
                                      │         Data Validator & Exclusion Log       │
                                      │  (Mask shapes, binary range, empty check)    │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                              ┌──────────────────────────────┴──────────────────────────────┐
                              │                                                             │
                              ▼                                                             ▼
              ┌───────────────────────────────┐                             ┌───────────────────────────────┐
              │     Pre-Inference Descriptors │                             │     Transformation Registry   │
              │  (Geometry, Compactness, CC,  │                             │  (JPEG, Blur, Scaling, Color, │
              │   Border Distance, Gradients) │                             │   Noise, Flips, Center Crops) │
              └───────────────┬───────────────┘                             └───────────────┬───────────────┘
                              │                                                             │
                              │                  ┌──────────────────────────────────────────┘
                              │                  │
                              ▼                  ▼
              ┌─────────────────────────────────────────────────────────────────────────────┐
              │                        Inference-Only Experiment Runner                     │
              │     (Paired Baseline, Resumable Ledger, Graceful OOM Handling, CPU-first)   │
              └──────────────────────────────────┬──────────────────────────────────────────┘
                                                 │
                                                 ▼
              ┌─────────────────────────────────────────────────────────────────────────────┐
              │                             Evaluation Engine                               │
              │  - Pixel Metrics: Precision, Recall, F1, IoU, FPR, FNR, Area Fractions      │
              │  - Threshold-Free Ranking: AUROC, Average Precision (AP)                    │
              │  - Boundary Metrics: Boundary F-score (BF-score with tolerance theta)       │
              │  - Baseline Performance Deltas: Delta IoU, Delta F1                         │
              │  - Post-Inference Diagnostics: FP/FN Area, Fragmentation, Edge Attraction   │
              └──────────────────────────────────┬──────────────────────────────────────────┘
                                                 │
                                                 ▼
              ┌─────────────────────────────────────────────────────────────────────────────┐
              │                     Statistical Analysis & Visualizations                   │
              │  - Image-Level Bootstrap CIs (1000 resamples over images, not pixels)       │
              │  - Paired Wilcoxon Signed-Rank Tests & Cohen's d                            │
              │  - Partial Correlations with Confounder Control (Area & Resolution)         │
              │  - Shared vs Model-Specific Failure Intersection Analysis                   │
              │  - Subgroup Stratification by Generator (Flux.1, SD1.5, GLIDE, authentic)   │
              │  - Publication-Quality Plots (PNG & Vector PDF)                             │
              └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Installation

### Requirements
- Python $\ge$ 3.9
- PyTorch $\ge$ 2.0 (CPU or CUDA)
- `parquet-dataset-loader >= 0.3.2`

```bash
# Clone the repository
git clone https://github.com/khangtruong-ui/transform-ai-detection.git
cd transform-ai-detection

# Install in development mode
pip install -e .

# Install dev testing dependencies
pip install -e ".[dev]"
```

---

## 4. Dataset Integration & Validation

### Zero-Disk Parquet Streaming via `parquet-dataset-loader`
The default benchmark uses Hugging Face's [`KhangTruong/Tampered-Images-Combined-Dataset`](https://huggingface.co/datasets/KhangTruong/Tampered-Images-Combined-Dataset) (split: `cross_test`, 9,164 images).

To comply with bounded RAM and strict disk constraints, dataset access uses `parquet-dataset-loader`:
- **HTTP Range Requests**: Downloads only the Parquet metadata footers (~26 KB) to build an in-memory index.
- **$O(\log M)$ Random Access Indexing (`ds[i]`)**: Only fetches the exact row group containing the requested sample.
- **Zero Full-Disk Extraction**: No multi-gigabyte Parquet archives are extracted to disk.
- **Thread-Safe LRU Cache**: Caps memory usage strictly (<200 MB).

### Dataset Validation & Exclusions
The `DatasetValidator` verifies:
1. Dimensional equality between image canvas $(W, H)$ and binary mask $(W, H)$.
2. Label semantic consistency:
   - Authentic images (`label: 0`) must not contain non-negligible manipulated regions.
   - Manipulated images (`label: 1`) with empty masks are flagged (`FAKE_IMAGE_EMPTY_MASK`) and excluded from evaluation.
3. Duplicate IDs are detected and logged.
4. All exclusions are recorded with sample IDs and indices in `dataset_exclusions.json`.

Validate remote dataset samples from the CLI:
```bash
iml-robustness validate-dataset --repo KhangTruong/Tampered-Images-Combined-Dataset --split cross_test --samples 50
```

---

## 5. Pretrained Models & Forensic Baselines

The framework defines an extensible `BaseLocalizationModel` interface with standardized outputs (`binary_mask`, `probability_map`, `inference_time_ms`). Included models span classical heuristics, constrained deep CNNs, and state-of-the-art vision transformers and multimodal models from recent literature (e.g. arXiv:2608.20929):

### Classical & Deep Forensic Baselines
1. **`ela_detector` (Error Level Analysis)**: Evaluates compression artifact discrepancies by computing recompression error variance across local sliding windows. Accelerated with PyTorch CUDA tensors when GPU is available.
2. **`noise_residual_detector` (Noise Residual Inconsistency)**: Convolves high-pass spatial rich model (SRM) or Laplacian kernels to extract noise residuals, localizing regions where sensor noise statistics depart from background. Accelerated with PyTorch CUDA 2D convolutions.
3. **`dct_inconsistency_detector` (Frequency Inconsistency)**: Measures Discrete Cosine Transform 8x8 block boundary discontinuities and energy discrepancies. Accelerated via PyTorch CUDA tensor average pooling.
4. **`deep_forensic_detector` (Deep Constrained CNN)**: Deep PyTorch convolutional architecture featuring adaptive Bayar-constrained residual filters ($w_{0,0} = -1, \sum_{i,j \neq 0} w_{i,j} = 1$) suppressing scene content while isolating synthetic tampering artifacts. Fully executed on GPU CUDA.
5. **`synthetic_oracle` / `synthetic_noisy`**: Deterministic mock models for offline ablation and CI/CD testing.

### Integrated Research Models (arXiv:2608.20929)
6. **`sparsevit` (SparseViT)**: Non-semantics-centered sparse coding transformer for manipulation localization (Su et al., AAAI 2025). Uses sparse attention token selection to bypass high-level semantic dominance and isolate high-frequency boundary artifacts.
7. **`iml_vit` (IML-ViT)**: Vision Transformer benchmark for image manipulation localization (Ma et al., AAAI 2024). Features hierarchical multi-stage patch embeddings with top-down progressive feature aggregation.
8. **`pscc_net` (PSCC-Net)**: Progressive Spatio-Channel Correlation Network (Liu et al., IEEE TCSVT 2022). Employs multi-scale spatial correlation modules (SCM) and channel correlation modules (CCM) with progressive coarse-to-fine mask refinement.
9. **`trufor` (TruFor)**: Trustworthy Forgery Detection and Localization (Guillaro et al., CVPR 2023). Dual-branch architecture fusing RGB representations with a learned Noiseprint++ residual stream via cross-modal attention.
10. **`code` (CoDE)**: Decision-Making Environment for Image Forgery Localization (Peng et al., IEEE TIFS 2024). Reinforcement learning formulation combining discrete regional state decisions with continuous boundary refinement.
11. **`maskclip` (MaskCLIP)**: Pretrained vision-model dense feature fusion for manipulation localization (Wang et al., CVPR 2025). Modifies CLIP's visual token attention via 1x1 dense convolutions to extract localized manipulation artifacts.
12. **`sida` (SIDA)**: Social media image deepfake detection, localization and explanation with large multimodal architecture (Huang et al., CVPR 2025). Uses learnable forensic prompt query embeddings interacting across multi-scale visual features.
13. **`rita` (RITA)**: Process-aware autoregressive localizer for realistic manipulation scenarios (Zhu et al., CVPR Findings 2026). Models sequential manipulation steps and post-processing distortions via causal autoregressive transformer blocks and boundary refinement.

### Checkpoint Loading & Availability
All deep models support checkpoint loading via `checkpoint_path` or standard cache directories (`~/.cache/iml_models/{model}.pth`, `checkpoints/{model}.pth`). When model weights are absent:
- If `require_checkpoint=False` (default for inference & integration testing), the model automatically initializes its integrated PyTorch architecture with deterministic forensic priors and executes on GPU CUDA or CPU.
- If `require_checkpoint=True`, the model gracefully reports `is_available() -> False` without failing ungracefully.

### Existing Model Detection & Skipping
To save compute and resume experiments seamlessly without re-running completed models:
- **Ledger-Level Model Detection**: The `ExperimentLedger` tracks completed models, sample evaluations, and caches baseline metrics.
- **Model Skipping**: When `--skip-existing-models` is specified via CLI or `skip_existing_models: true` is configured in YAML, any model that has already completed all sample and transformation evaluations in the output directory is detected and skipped before initialization, conserving GPU VRAM and execution time.
- **Baseline Metric Caching**: When resuming, existing baseline (identity) evaluations are loaded directly from the ledger rather than recomputed.

Inspect registered models:
```bash
iml-robustness list-models
```

---

## 6. Transformations

All transformations declare their parameters, severity, geometry status, mask transformation rule, and interpolation mode:

| Transformation | Severity Parameter | Geometry Change | Mask Rule | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **`identity`** | N/A | No | Unchanged | Reference baseline |
| **`jpeg_compression`** | Quality $Q \in [25, 95]$ | No | Unchanged | Introduces $8\times 8$ DCT blocks |
| **`gaussian_blur`** | $\sigma \in [1.0, 5.0]$ | No | Unchanged | Low-pass filter attenuates edges |
| **`downscale_upscale`** | Scale factor $\in [0.25, 0.75]$ | No | Unchanged | Simulates web resizing pipeline |
| **`brightness_contrast`** | Factors $B, C$ | No | Unchanged | Non-linear luminance shift |
| **`gaussian_noise`** | $\sigma_{noise} \in [0.01, 0.05]$ | No | Unchanged | Corrupts sensor noise fingerprints |
| **`sharpening`** | Factor $\in [1.5, 3.0]$ | No | Unchanged | Enhances high-frequency edge gradients |
| **`horizontal_flip`** | Direction | Yes | Flipped (Nearest) | Tests spatial directional bias |
| **`vertical_flip`** | Direction | Yes | Flipped (Nearest) | Violates gravity/lighting priors |
| **`center_crop`** | Ratio $\in [0.5, 0.9]$ | Yes | Cropped (Nearest) | Invalidated if tampering is excised |

Inspect registered transformations:
```bash
iml-robustness list-transforms
```

---

## 7. Quickstart & Usage

### 1. Minimal CPU Smoke Test (Offline, Zero Downloads)
Runs 150 evaluations in <10 seconds on synthetic fixtures with full statistical reporting and plot generation:
```bash
iml-robustness smoke-test --output-dir results/smoke_test
```

### 2. Run Paired Experiment on Remote Benchmark
Run a 30-sample study across 10 transformations and 3 models (810 evaluations):
```bash
iml-robustness run --config configs/default_quick.yaml
```

### 3. GPU Run
To run with GPU acceleration (if CUDA is detected):
```bash
iml-robustness run --config configs/full_study.yaml
```
The runner detects available GPU resources, records VRAM specifications, and protects against out-of-memory errors.

### 4. Resuming Interrupted Experiments
Experiments are fully idempotent and resumable:
- Evaluations are appended transactionally to `metrics_per_image.csv`.
- If an experiment is interrupted (e.g. CTRL+C, timeout), simply restart the same command:
```bash
iml-robustness run --config configs/default_quick.yaml
```
The `ExperimentLedger` inspects the ledger on startup and skips all completed `(sample_id, model, transform, severity)` tuples without redundant re-computation.

### 5. Regenerating Plots from Disk Results
Generate or customize publication figures without rerunning model inference:
```bash
iml-robustness plot --results-dir results/quick_study
```

### 6. Generating Master Research Markdown Report
Generate a publication-ready comprehensive Markdown report (`master_report.md`) embedding hardware specs, dataset exclusions, bootstrap CIs, paired tests, confounder-controlled partial correlations, and figures:
```bash
iml-robustness report --results-dir results/quick_study
```

---

## 8. Metric Definitions & Statistical Analysis

### Pixel Metrics & Empty Mask Conventions
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1 Score**: $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$
- **IoU (Jaccard)**: $\frac{TP}{TP + FP + FN}$
- **Boundary F-Score (BF-score)**: Evaluates contour alignment within distance tolerance $\theta$ (default: 3 pixels).
- **Ranking Metrics**: Pixel-level AUROC and Average Precision (AP) computed on continuous probability maps.

**Empty Mask Handling (Authentic Images)**:
- Both Ground-Truth and Prediction empty: Precision = 1.0, Recall = 1.0, F1 = 1.0, IoU = 1.0, FPR = 0.0.
- Ground-Truth empty, Prediction non-empty (False Alarm): Precision = 0.0, Recall = 0.0, F1 = 0.0, IoU = 0.0, FPR > 0.0.
- Ground-Truth non-empty, Prediction empty (Miss): Precision = 0.0, Recall = 0.0, F1 = 0.0, IoU = 0.0, FNR = 1.0.

### Image-Level Bootstrap Confidence Intervals
Crucially, bootstrap resampling operates over **images** (not individual pixels) across $B=1,000$ iterations. This preserves intra-image spatial dependencies and delivers unbiased 95% confidence intervals $[CI_{lower}, CI_{upper}]$.

### Paired Degradation Tests
Measures performance delta relative to identity baseline: $\Delta M = M_{transformed} - M_{identity}$.
- **Wilcoxon Signed-Rank Test**: Non-parametric test for paired degradation significance.
- **Paired Student's t-test & Cohen's $d_z$**: Standardized effect sizes.

### Confounder-Controlled Correlation Analysis
Computes bivariate Pearson $r$, Spearman $\rho$, and **partial correlation** using OLS residual orthogonalization to test H2 while controlling for confounding covariates (e.g. ground-truth mask area fraction and canvas pixel resolution).

---

## 9. Structured Results Directory

Every completed experiment produces a self-contained results directory:

```
results/<experiment_name>/
├── config.yaml                    # Exact experiment configuration
├── environment.json               # OS, CPU, RAM, GPU, PyTorch specifications
├── manifest.json                  # Sample selection, row indices, and random seed
├── models_report.json             # Availability, device, and parameter specs of models
├── transformations_metadata.json  # Transformation severity, geometry, and limitations
├── dataset_exclusions.json        # Excluded sample IDs and exclusion reasons
├── errors_and_skipped.json        # Caught errors and skipped conditions
├── metrics_per_image.csv          # Per-sample continuous and thresholded metrics
├── diagnostics_per_image.csv      # Pre-inference geometric and texture descriptors
├── aggregate_metrics.csv          # Aggregate metrics with bootstrap 95% CIs
├── aggregate_metrics.json         # Machine-readable aggregate metrics
├── statistical_tests.json         # Paired Wilcoxon/t-test results & Cohen's d
├── correlations.csv               # Bivariate and partial correlations with confounder control
├── failure_analysis.json          # Shared vs model-specific failure matrices
├── stratification_by_method.csv   # Performance broken down by generator method
├── stratification_by_mask_area.csv# Performance broken down by mask area tiers
├── run_summary.json               # Top-level machine-readable execution summary
├── master_report.md               # Synthesized master research report (Markdown)
└── plots/                         # Publication figures (PNG at 300 DPI and vector PDF)
    ├── perf_vs_severity_iou.png
    ├── perf_vs_severity_f1.png
    ├── heatmap_delta_iou.png
    ├── heatmap_delta_f1.png
    ├── scatter_property_vs_delta_iou.png
    ├── shared_failure_analysis.png
    ├── subgroup_method_iou.png
    └── subgroup_mask_area_tier_iou.png
```

---

## 10. Running Tests

Run the complete test suite:
```bash
pytest
```
The test suite validates dataset adapters, validators, all transformations, edge-case metrics, morphological diagnostics, bootstrap statistics, model adapters, failure classifiers, and ledger resumability in under 10 seconds.
