"""
experiments/run_baselines.py
-----------------------------

Fresh baseline experiment runner.

Every execution regenerates all baseline results from scratch.
Existing JSON result files are overwritten.

Models:
    - static_gcn
    - tgat
    - tgn
    - tgnn
    - cnn
    - isolation_forest

Features:
    - Always fresh: existing results are overwritten.
    - CUDA-safe.
    - CPU-only handling for Isolation Forest.
    - Immediate result saving after every model.
    - One model failure does not stop the remaining experiments.
    - Detailed logging.
    - Proper __main__ entry point.

Usage:

    python experiments/run_baselines.py

    python experiments/run_baselines.py --seeds 42 123 2024

    python experiments/run_baselines.py \
        --seeds 42 123 2024 3407 7777 \
        --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger(__name__)


# ============================================================================
# RANDOM SEED
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# JSON SAVING
# ============================================================================

def save_json(path: Path, result: dict) -> None:
    """
    Save result dictionary to JSON.

    Existing files are intentionally overwritten.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )


# ============================================================================
# CNN BASELINE
# ============================================================================

def run_cnn_baseline(
    train_s,
    val_s,
    test_s,
    device,
    cfg,
    seed,
):
    """
    CNN baseline.

    Each temporal snapshot is converted into a single feature vector
    by mean-pooling node features from every node type.

    The CNN predicts one fraud score per snapshot.

    That snapshot-level score is then broadcast to all events
    belonging to the snapshot for event-level evaluation.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    set_seed(seed)

    # ------------------------------------------------------------------------
    # Snapshot -> matrix
    # ------------------------------------------------------------------------

    def to_matrix(snaps):

        features = []
        labels = []

        for snapshot in snaps:

            node_vectors = []

            for node_type in snapshot.node_types:

                if "x" not in snapshot[node_type]:
                    continue

                x = snapshot[node_type].x

                if x.shape[0] == 0:
                    continue

                x_cpu = x.detach().cpu()

                node_vectors.append(
                    x_cpu.mean(dim=0)
                )

            if not node_vectors:
                raise RuntimeError(
                    "CNN encountered a snapshot with no node features."
                )

            row = torch.cat(
                node_vectors,
                dim=0,
            )

            features.append(row)

            labels.append(
                snapshot.event_labels
                .detach()
                .cpu()
            )

        if not features:
            raise RuntimeError(
                "CNN received an empty snapshot list."
            )

        return (
            torch.stack(features),
            labels,
        )

    # ------------------------------------------------------------------------
    # Build training matrix
    # ------------------------------------------------------------------------

    train_x, _ = to_matrix(train_s)

    input_dim = train_x.shape[1]

    log.info(
        "CNN feature dimension: %d",
        input_dim,
    )

    # ------------------------------------------------------------------------
    # CNN model
    # ------------------------------------------------------------------------

    class CNN1D(nn.Module):

        def __init__(self, in_features: int):
            super().__init__()

            self.conv = nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            )

            self.fc = nn.Sequential(
                nn.Linear(
                    32 * in_features,
                    64,
                ),
                nn.ReLU(),
                nn.Linear(
                    64,
                    1,
                ),
            )

        def forward(self, x):

            # (T, F)
            x = x.unsqueeze(1)

            # (T, 1, F)
            x = F.relu(
                self.conv(x)
            )

            # (T, 32, F)
            x = x.reshape(
                x.shape[0],
                -1,
            )

            # (T,)
            return torch.sigmoid(
                self.fc(x)
            ).squeeze(-1)

    model = CNN1D(
        input_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.BCELoss()

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    training_start = time.perf_counter()

    model.train()

    epochs = 50

    for epoch in range(
        1,
        epochs + 1,
    ):

        optimizer.zero_grad()

        x_train, _ = to_matrix(
            train_s
        )

        y_train = torch.tensor(
            [
                float(
                    snapshot.event_labels
                    .detach()
                    .float()
                    .mean()
                    .item()
                )
                for snapshot in train_s
            ],
            dtype=torch.float32,
        )

        x_train = x_train.to(device)
        y_train = y_train.to(device)

        predictions = model(
            x_train
        )

        loss = criterion(
            predictions,
            y_train,
        )

        loss.backward()

        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            log.info(
                "  CNN epoch %d/%d | loss=%.6f",
                epoch,
                epochs,
                loss.item(),
            )

    training_time = (
        time.perf_counter()
        - training_start
    )

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------

    model.eval()

    inference_start = time.perf_counter()

    all_true = []
    all_scores = []

    with torch.no_grad():

        x_test, test_labels = to_matrix(
            test_s
        )

        x_test = x_test.to(device)

        snapshot_scores = model(
            x_test
        )

        snapshot_scores = (
            snapshot_scores
            .detach()
            .cpu()
            .numpy()
        )

        for score, labels in zip(
            snapshot_scores,
            test_labels,
        ):

            labels_np = (
                labels
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            all_true.extend(
                labels_np
            )

            all_scores.extend(
                [float(score)] * len(labels_np)
            )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    y_true = np.asarray(
        all_true,
        dtype=int,
    )

    y_score = np.asarray(
        all_scores,
        dtype=float,
    )

    y_pred = (
        y_score >= 0.5
    ).astype(int)

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if len(np.unique(y_true)) > 1:

        roc_auc = roc_auc_score(
            y_true,
            y_score,
        )

    else:

        roc_auc = 0.0

    return {
        "test_precision": float(
            precision
        ),
        "test_recall": float(
            recall
        ),
        "test_f1": float(
            f1
        ),
        "test_roc_auc": float(
            roc_auc
        ),
        "test_training_time_s": float(
            training_time
        ),
        "test_inference_time_s": float(
            inference_time
        ),
    }


# ============================================================================
# ISOLATION FOREST
# ============================================================================

def run_isolation_forest(
    train_s,
    test_s,
    seed,
):
    """
    Isolation Forest baseline.

    All data passed to sklearn is explicitly converted to CPU NumPy arrays.
    """

    from sklearn.ensemble import IsolationForest

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    # ------------------------------------------------------------------------
    # Snapshot feature extraction
    # ------------------------------------------------------------------------

    def snapshot_features(snaps):

        rows = []

        for snapshot in snaps:

            node_vectors = []

            for node_type in snapshot.node_types:

                if "x" not in snapshot[node_type]:
                    continue

                x = snapshot[node_type].x

                if x.shape[0] == 0:
                    continue

                x_cpu = (
                    x.detach()
                    .cpu()
                )

                node_vectors.append(
                    x_cpu.mean(dim=0)
                )

            if not node_vectors:

                raise RuntimeError(
                    "Isolation Forest encountered "
                    "a snapshot with no node features."
                )

            row = np.concatenate(
                [
                    tensor.numpy()
                    for tensor in node_vectors
                ]
            )

            rows.append(row)

        if not rows:

            raise RuntimeError(
                "Isolation Forest received an empty snapshot list."
            )

        return np.stack(rows)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------

    log.info(
        "Preparing Isolation Forest training data..."
    )

    X_train = snapshot_features(
        train_s
    )

    log.info(
        "Isolation Forest train shape: %s",
        X_train.shape,
    )

    X_test = snapshot_features(
        test_s
    )

    log.info(
        "Isolation Forest test shape: %s",
        X_test.shape,
    )

    # ------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------

    training_start = time.perf_counter()

    clf = IsolationForest(
        random_state=seed,
        contamination=0.10,
    )

    clf.fit(
        X_train
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------

    inference_start = time.perf_counter()

    scores = -clf.score_samples(
        X_test
    )

    snapshot_predictions = (
        clf.predict(X_test) == -1
    ).astype(int)

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # ------------------------------------------------------------------------
    # Event-level broadcasting
    # ------------------------------------------------------------------------

    all_true = []
    all_scores = []
    all_predictions = []

    for i, snapshot in enumerate(
        test_s
    ):

        labels = (
            snapshot.event_labels
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        all_true.extend(
            labels
        )

        all_scores.extend(
            [float(scores[i])] * len(labels)
        )

        all_predictions.extend(
            [
                int(snapshot_predictions[i])
            ] * len(labels)
        )

    y_true = np.asarray(
        all_true,
        dtype=int,
    )

    y_score = np.asarray(
        all_scores,
        dtype=float,
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=int,
    )

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if len(np.unique(y_true)) > 1:

        roc_auc = roc_auc_score(
            y_true,
            y_score,
        )

    else:

        roc_auc = 0.0

    return {
        "test_precision": float(
            precision
        ),
        "test_recall": float(
            recall
        ),
        "test_f1": float(
            f1
        ),
        "test_roc_auc": float(
            roc_auc
        ),
        "test_training_time_s": float(
            training_time
        ),
        "test_inference_time_s": float(
            inference_time
        ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    # ------------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Run all baseline experiments "
            "from scratch."
        )
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            42,
            123,
            2024,
            3407,
            7777,
        ],
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )

    parser.add_argument(
        "--data-dir",
        default="data/synthetic/",
    )

    parser.add_argument(
        "--out-dir",
        default="results/raw/baselines/",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # START
    # ------------------------------------------------------------------------

    experiment_start = time.perf_counter()

    log.info("=" * 80)
    log.info(
        "STARTING FRESH BASELINE EXPERIMENTS"
    )
    log.info("=" * 80)

    log.info(
        "Seeds: %s",
        args.seeds,
    )

    log.info(
        "Config: %s",
        args.config,
    )

    log.info(
        "Data: %s",
        args.data_dir,
    )

    log.info(
        "Output: %s",
        args.out_dir,
    )

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------------

    import torch

    from data.temporal_dataset import (
        TemporalCredentialDataset,
    )

    from graph.snapshots import (
        chronological_split,
    )

    from training.train import (
        build_model,
        train_one_epoch,
        compute_class_weight,
    )

    from training.evaluate import (
        evaluate,
    )

    from training.early_stopping import (
        EarlyStopping,
    )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log.info(
        "Using device: %s",
        device,
    )

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    out_path = Path(
        args.out_dir
    )

    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    log.info(
        "Loading dataset from %s",
        args.data_dir,
    )

    dataset = TemporalCredentialDataset(
        args.data_dir,
        cfg["dataset"]["n_snapshots"],
    )

    snapshots = list(
        dataset
    )

    input_dims = (
        dataset.get_input_dims()
    )

    log.info(
        "Dataset loaded: %d snapshots",
        len(snapshots),
    )

    # ------------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------------

    train_ids, val_ids, test_ids = (
        chronological_split(
            cfg["dataset"]["n_snapshots"],
            cfg["split"]["train_ratio"],
            cfg["split"]["val_ratio"],
        )
    )

    train_s = [
        snapshots[i]
        for i in train_ids
    ]

    val_s = [
        snapshots[i]
        for i in val_ids
    ]

    test_s = [
        snapshots[i]
        for i in test_ids
    ]

    log.info(
        "Split: train=%d | validation=%d | test=%d snapshots",
        len(train_s),
        len(val_s),
        len(test_s),
    )

    # ------------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------------

    tcfg = cfg["training"]

    # ========================================================================
    # SEED LOOP
    # ========================================================================

    for seed in args.seeds:

        seed_start = time.perf_counter()

        log.info("")
        log.info("=" * 80)
        log.info(
            "STARTING SEED %d",
            seed,
        )
        log.info("=" * 80)

        set_seed(seed)

        # --------------------------------------------------------------------
        # Class weight
        # --------------------------------------------------------------------

        log.info(
            "Computing class weight for seed %d...",
            seed,
        )

        pos_w = compute_class_weight(
            train_s,
            cfg,
        )

        # ====================================================================
        # GNN MODELS
        # ====================================================================

        for model_name in [
            "static_gcn",
            "tgat",
            "tgn",
            "tgnn",
        ]:

            result_file = (
                out_path
                / f"{model_name}_seed_{seed}.json"
            )

            log.info("")
            log.info(
                "-" * 70
            )
            log.info(
                "STARTING %s | seed=%d",
                model_name,
                seed,
            )
            log.info(
                "Result will be written to: %s",
                result_file,
            )
            log.info(
                "-" * 70
            )

            model_start = time.perf_counter()

            try:

                set_seed(seed)

                # ------------------------------------------------------------
                # Build model
                # ------------------------------------------------------------

                model = build_model(
                    model_name,
                    input_dims,
                    cfg,
                ).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=tcfg["learning_rate"],
                )

                early_stopping = EarlyStopping(
                    patience=tcfg[
                        "early_stopping_patience"
                    ],
                )

                # ------------------------------------------------------------
                # Training
                # ------------------------------------------------------------

                for epoch in range(
                    1,
                    tcfg["epochs"] + 1,
                ):

                    train_one_epoch(
                        model,
                        train_s,
                        optimizer,
                        device,
                        pos_w,
                    )

                    validation_metrics = (
                        evaluate(
                            model,
                            val_s,
                            device,
                        )
                    )

                    should_stop = (
                        early_stopping(
                            1
                            - validation_metrics[
                                "f1"
                            ]
                        )
                    )

                    if epoch == 1 or epoch % 10 == 0:

                        log.info(
                            "  %s | epoch=%d | val_f1=%.4f",
                            model_name,
                            epoch,
                            validation_metrics[
                                "f1"
                            ],
                        )

                    if should_stop:

                        log.info(
                            "  %s | early stopping at epoch %d",
                            model_name,
                            epoch,
                        )

                        break

                # ------------------------------------------------------------
                # Test
                # ------------------------------------------------------------

                test_metrics = evaluate(
                    model,
                    test_s,
                    device,
                )

                model_time = (
                    time.perf_counter()
                    - model_start
                )

                result = {
                    "model": model_name,
                    "seed": seed,
                    **{
                        f"test_{key}": value
                        for key, value
                        in test_metrics.items()
                    },
                    "total_runtime_s": float(
                        model_time
                    ),
                }

                # ------------------------------------------------------------
                # ALWAYS OVERWRITE RESULT
                # ------------------------------------------------------------

                save_json(
                    result_file,
                    result,
                )

                log.info(
                    "COMPLETED %s | seed=%d | F1=%.4f | AUC=%.4f | %.2f min",
                    model_name,
                    seed,
                    test_metrics["f1"],
                    test_metrics["roc_auc"],
                    model_time / 60,
                )

                # ------------------------------------------------------------
                # Cleanup
                # ------------------------------------------------------------

                del model
                del optimizer
                del early_stopping

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception:

                log.exception(
                    "FAILED %s | seed=%d",
                    model_name,
                    seed,
                )

                log.info(
                    "Continuing with the next model..."
                )

                continue

        # ====================================================================
        # CNN
        # ====================================================================

        cnn_file = (
            out_path
            / f"cnn_seed_{seed}.json"
        )

        log.info("")
        log.info(
            "-" * 70
        )
        log.info(
            "STARTING CNN | seed=%d",
            seed,
        )
        log.info(
            "Result will be written to: %s",
            cnn_file,
        )
        log.info(
            "-" * 70
        )

        try:

            cnn_result = run_cnn_baseline(
                train_s,
                val_s,
                test_s,
                device,
                cfg,
                seed,
            )

            save_json(
                cnn_file,
                {
                    "model": "cnn",
                    "seed": seed,
                    **cnn_result,
                },
            )

            log.info(
                "COMPLETED CNN | seed=%d | F1=%.4f | AUC=%.4f",
                seed,
                cnn_result["test_f1"],
                cnn_result["test_roc_auc"],
            )

        except Exception:

            log.exception(
                "FAILED CNN | seed=%d",
                seed,
            )

            log.info(
                "Continuing with Isolation Forest..."
            )

        # ====================================================================
        # ISOLATION FOREST
        # ====================================================================

        iso_file = (
            out_path
            / f"isolation_forest_seed_{seed}.json"
        )

        log.info("")
        log.info(
            "-" * 70
        )
        log.info(
            "STARTING ISOLATION FOREST | seed=%d",
            seed,
        )
        log.info(
            "Result will be written to: %s",
            iso_file,
        )
        log.info(
            "-" * 70
        )

        try:

            iso_result = run_isolation_forest(
                train_s,
                test_s,
                seed,
            )

            save_json(
                iso_file,
                {
                    "model": "isolation_forest",
                    "seed": seed,
                    **iso_result,
                },
            )

            log.info(
                "COMPLETED Isolation Forest | seed=%d | F1=%.4f | AUC=%.4f",
                seed,
                iso_result["test_f1"],
                iso_result["test_roc_auc"],
            )

        except Exception:

            log.exception(
                "FAILED Isolation Forest | seed=%d",
                seed,
            )

        # ====================================================================
        # SEED COMPLETE
        # ====================================================================

        seed_time = (
            time.perf_counter()
            - seed_start
        )

        log.info("")
        log.info(
            "=" * 80
        )
        log.info(
            "COMPLETED SEED %d | %.2f min",
            seed,
            seed_time / 60,
        )
        log.info(
            "=" * 80
        )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    total_time = (
        time.perf_counter()
        - experiment_start
    )

    log.info("")
    log.info("=" * 80)
    log.info(
        "ALL BASELINE EXPERIMENTS COMPLETE"
    )
    log.info(
        "Total runtime: %.2f minutes",
        total_time / 60,
    )
    log.info(
        "Results directory: %s",
        out_path,
    )
    log.info("=" * 80)


# ============================================================================
# IMPORTANT: EXECUTE MAIN WHEN RUN AS A SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()