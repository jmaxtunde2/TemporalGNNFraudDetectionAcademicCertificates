"""
training/evaluate.py
---------------------
Evaluation utilities.  Computes precision, recall, F1, ROC-AUC,
and inference time from model outputs.
"""
from __future__ import annotations
import time
import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)
from torch_geometric.data import HeteroData


def evaluate(
    model: torch.nn.Module,
    snapshots: list[HeteroData],
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Run inference on a list of snapshots and return metrics.

    Returns
    -------
    dict with keys: precision, recall, f1, roc_auc, inference_time_s
    """
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[int]   = []

    t0 = time.perf_counter()
    with torch.no_grad():
        probs_list, labels_list = model(snapshots, device)
    inference_time = time.perf_counter() - t0

    for probs, labels in zip(probs_list, labels_list):
        if probs.numel() == 0:
            continue
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    if len(all_labels) == 0:
        return dict(precision=0., recall=0., f1=0., roc_auc=0.,
                    inference_time_s=inference_time)

    y_true  = np.array(all_labels)
    y_score = np.array(all_probs)
    y_pred  = (y_score >= threshold).astype(int)

    metrics = dict(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)
                      if len(np.unique(y_true)) > 1 else 0.),
        inference_time_s=inference_time,
    )
    return metrics
