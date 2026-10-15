"""Pixel-level evaluation metrics with explicit empty-mask handling."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import numpy as np


@dataclass
class PixelMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    fpr: float
    fnr: float
    gt_area_fraction: float
    pred_area_fraction: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        return {k: float(v) for k, v in d.items()}


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute (TP, FP, TN, FN) for binary masks."""
    y_true_b = (y_true > 0).astype(bool)
    y_pred_b = (y_pred > 0).astype(bool)

    tp = int(np.logical_and(y_true_b, y_pred_b).sum())
    fp = int(np.logical_and(~y_true_b, y_pred_b).sum())
    tn = int(np.logical_and(~y_true_b, ~y_pred_b).sum())
    fn = int(np.logical_and(y_true_b, ~y_pred_b).sum())

    return tp, fp, tn, fn


def compute_pixel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> PixelMetrics:
    """Compute pixel-level classification and segmentation metrics.

    Empty Mask Convention:
    ---------------------
    - Both y_true and y_pred are empty (all 0s):
      Precision=1.0, Recall=1.0, F1=1.0, IoU=1.0, FPR=0.0, FNR=0.0.
      (Authentic sample correctly localized with no false positives).
    - y_true is empty (0s) but y_pred contains positive pixels:
      Precision=0.0, Recall=0.0, F1=0.0, IoU=0.0, FPR=FP/(FP+TN), FNR=0.0.
      (False alarms on authentic image).
    - y_true has positive pixels but y_pred is empty (0s):
      Precision=0.0, Recall=0.0, F1=0.0, IoU=0.0, FPR=0.0, FNR=1.0.
      (Manipulated region completely missed).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    tp, fp, tn, fn = compute_confusion_matrix(y_true, y_pred)
    total_pixels = y_true.size

    gt_area_fraction = float((tp + fn) / total_pixels) if total_pixels > 0 else 0.0
    pred_area_fraction = float((tp + fp) / total_pixels) if total_pixels > 0 else 0.0

    # Case 1: Both empty
    if (tp + fn) == 0 and (tp + fp) == 0:
        return PixelMetrics(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            iou=1.0,
            fpr=0.0,
            fnr=0.0,
            gt_area_fraction=gt_area_fraction,
            pred_area_fraction=pred_area_fraction,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    # Case 2: GT is empty, pred is non-empty
    if (tp + fn) == 0 and (tp + fp) > 0:
        neg_denom = fp + tn
        fpr = float(fp / neg_denom) if neg_denom > 0 else 0.0
        return PixelMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            iou=0.0,
            fpr=fpr,
            fnr=0.0,
            gt_area_fraction=gt_area_fraction,
            pred_area_fraction=pred_area_fraction,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    # Case 3: GT is non-empty, pred is empty
    if (tp + fn) > 0 and (tp + fp) == 0:
        return PixelMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            iou=0.0,
            fpr=0.0,
            fnr=1.0,
            gt_area_fraction=gt_area_fraction,
            pred_area_fraction=pred_area_fraction,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    # Standard case: GT non-empty, pred non-empty
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    neg_denom = fp + tn
    fpr = float(fp / neg_denom) if neg_denom > 0 else 0.0

    pos_denom = fn + tp
    fnr = float(fn / pos_denom) if pos_denom > 0 else 0.0

    return PixelMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        fpr=fpr,
        fnr=fnr,
        gt_area_fraction=gt_area_fraction,
        pred_area_fraction=pred_area_fraction,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )
