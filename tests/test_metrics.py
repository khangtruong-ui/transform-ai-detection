"""Unit tests for pixel metrics, boundary metrics, and ranking metrics."""

from __future__ import annotations

import numpy as np
import pytest

from transform_ai_detection.metrics.pixel import (
    compute_confusion_matrix,
    compute_pixel_metrics,
)
from transform_ai_detection.metrics.boundary import (
    compute_boundary_metrics,
    extract_boundary,
)
from transform_ai_detection.metrics.ranking import compute_ranking_metrics
from transform_ai_detection.metrics.engine import EvaluationEngine


def test_confusion_matrix_and_pixel_metrics():
    # 4x4 known masks
    y_true = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)

    y_pred = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)

    # TP = 3, FP = 1, TN = 11, FN = 1
    tp, fp, tn, fn = compute_confusion_matrix(y_true, y_pred)
    assert tp == 3
    assert fp == 1
    assert tn == 11
    assert fn == 1

    m = compute_pixel_metrics(y_true, y_pred)
    assert m.precision == 3 / 4  # 0.75
    assert m.recall == 3 / 4     # 0.75
    assert m.f1 == 0.75
    assert m.iou == 3 / 5        # 0.60


def test_empty_mask_edge_cases():
    # Case 1: Both GT and Pred are all-black (authentic image correctly localized)
    empty = np.zeros((32, 32), dtype=np.uint8)
    m1 = compute_pixel_metrics(empty, empty)
    assert m1.precision == 1.0
    assert m1.recall == 1.0
    assert m1.f1 == 1.0
    assert m1.iou == 1.0
    assert m1.fpr == 0.0

    # Case 2: GT is empty, pred has false alarms
    pred_bad = np.zeros((32, 32), dtype=np.uint8)
    pred_bad[0:5, 0:5] = 1
    m2 = compute_pixel_metrics(empty, pred_bad)
    assert m2.precision == 0.0
    assert m2.recall == 0.0
    assert m2.f1 == 0.0
    assert m2.iou == 0.0
    assert m2.fpr > 0.0

    # Case 3: GT has tampering, pred missed completely (all-black)
    gt_pos = np.zeros((32, 32), dtype=np.uint8)
    gt_pos[0:5, 0:5] = 1
    m3 = compute_pixel_metrics(gt_pos, empty)
    assert m3.precision == 0.0
    assert m3.recall == 0.0
    assert m3.f1 == 0.0
    assert m3.iou == 0.0
    assert m3.fnr == 1.0


def test_boundary_metrics():
    # Perfect alignment
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 1
    b_res = compute_boundary_metrics(mask, mask, tolerance=2)
    assert b_res.boundary_precision == 1.0
    assert b_res.boundary_recall == 1.0
    assert b_res.boundary_f1 == 1.0

    # Displaced by 1 pixel (should match within tolerance 2)
    mask_shift = np.zeros((64, 64), dtype=np.uint8)
    mask_shift[21:45, 20:44] = 1
    b_res_tol = compute_boundary_metrics(mask, mask_shift, tolerance=2)
    assert b_res_tol.boundary_f1 > 0.8

    # Empty masks boundary case
    empty = np.zeros((64, 64), dtype=np.uint8)
    b_empty = compute_boundary_metrics(empty, empty, tolerance=2)
    assert b_empty.boundary_f1 == 1.0


def test_ranking_metrics_auroc_and_ap():
    y_true = np.zeros((32, 32), dtype=np.uint8)
    y_true[10:20, 10:20] = 1

    # Perfect probability map
    y_prob_perfect = y_true.astype(np.float32) * 0.9 + 0.05
    auroc, ap = compute_ranking_metrics(y_true, y_prob_perfect)
    assert auroc == 1.0
    assert ap == 1.0

    # Inverted probability map
    y_prob_inverted = (1 - y_true).astype(np.float32)
    auroc_inv, _ = compute_ranking_metrics(y_true, y_prob_inverted)
    assert auroc_inv == 0.0

    # Single-class edge case (e.g. authentic image) -> undefined
    empty = np.zeros((32, 32), dtype=np.uint8)
    auroc_empty, ap_empty = compute_ranking_metrics(empty, y_prob_perfect)
    assert auroc_empty is None
    assert ap_empty is None


def test_evaluation_engine_deltas():
    engine = EvaluationEngine()
    y_true = np.zeros((32, 32), dtype=np.uint8)
    y_true[10:20, 10:20] = 1

    pred_base = np.copy(y_true)
    pred_trans = np.zeros((32, 32), dtype=np.uint8)  # degraded to zero

    base_m = engine.evaluate_sample(y_true, pred_base)
    trans_m = engine.evaluate_sample(y_true, pred_trans)

    deltas = engine.compute_deltas(base_m, trans_m)
    assert deltas["delta_iou"] == -1.0
    assert deltas["delta_f1"] == -1.0
