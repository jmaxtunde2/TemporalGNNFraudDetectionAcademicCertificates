"""Evaluation utilities with auditable raw predictions and timing/memory capture."""
from __future__ import annotations
import time
import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from torch_geometric.data import HeteroData


def evaluate(
    model: torch.nn.Module,
    snapshots: list[HeteroData],
    device: torch.device,
    threshold: float = 0.5,
    return_predictions: bool = False,
    prediction_output: str | None = None,
    split_name: str = "test",
) -> dict:
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    snapshot_ids: list[int] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs_list, labels_list = model(snapshots, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - t0

    for local_sid, (probs, labels) in enumerate(zip(probs_list, labels_list)):
        if probs.numel() == 0:
            continue
        n = probs.numel()
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
        global_sid = int(getattr(snapshots[local_sid], "snapshot_id", local_sid))
        snapshot_ids.extend([global_sid] * n)

    if len(all_labels) == 0:
        return dict(precision=0., recall=0., f1=0., roc_auc=0.,
                    inference_time_s=inference_time,
                    inference_samples_per_s=0., peak_gpu_memory_bytes=0,
                    n_samples=0)

    y_true = np.asarray(all_labels, dtype=int)
    y_score = np.asarray(all_probs, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    fpr, tpr, thresholds = roc_curve(y_true, y_score) if len(np.unique(y_true)) > 1 else (np.array([]), np.array([]), np.array([]))

    result = dict(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        inference_time_s=float(inference_time),
        inference_samples_per_s=float(len(y_true) / inference_time) if inference_time > 0 else 0.,
        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        n_samples=int(len(y_true)),
        threshold=float(threshold),
        confusion_matrix=cm,
        classification_report=report,
    )

    if return_predictions or prediction_output:
        pred_path = prediction_output
        if pred_path:
            from pathlib import Path
            Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd
            pd.DataFrame({
                "split": split_name,
                "snapshot_id": snapshot_ids,
                "y_true": y_true,
                "y_score": y_score,
                "y_pred": y_pred,
            }).to_csv(pred_path, index=False)
        result["prediction_file"] = pred_path
        result["roc_curve"] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()
        }

    return result
