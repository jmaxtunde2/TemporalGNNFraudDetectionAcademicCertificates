"""
experiments/run_baselines.py
-----------------------------
Trains all baseline models over all configured seeds.

Models: static_gcn, tgat, tgn, tgnn (proposed)
Also includes: CNN (1-D conv over snapshot features) and
               Isolation Forest (sklearn, no gradient).

Usage:
    python experiments/run_baselines.py
    python experiments/run_baselines.py --seeds 42 123 2024
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_cnn_baseline(train_s, val_s, test_s, device, cfg, seed):
    """
    CNN baseline: treats snapshots as a 1-D temporal sequence of
    mean node-feature vectors, then applies 1-D convolution.
    """
    import torch, torch.nn as nn, torch.nn.functional as F
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Build feature matrix: (n_snapshots, feature_dim)
    def to_matrix(snaps):
        feats, labels = [], []
        for s in snaps:
            row = torch.cat([s[nt].x.mean(0) for nt in s.node_types
                             if nt in s.node_types], dim=0)
            feats.append(row)
            labels.append(s.event_labels)
        return torch.stack(feats), labels  # (T, F), list

    # Simple 1-D CNN model
    class CNN1D(nn.Module):
        def __init__(self, in_f):
            super().__init__()
            self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.fc   = nn.Sequential(nn.Linear(32 * in_f, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, x):   # x: (T, F)
            x = x.unsqueeze(1)  # (T,1,F)
            x = F.relu(self.conv(x)).reshape(x.shape[0], -1)
            return torch.sigmoid(self.fc(x)).squeeze(-1)

    all_s = train_s + val_s + test_s
    mat, _ = to_matrix(all_s)
    in_f   = mat.shape[1]
    model  = CNN1D(in_f).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train epochs
    t0 = time.perf_counter()
    for _ in range(50):
        model.train(); opt.zero_grad()
        x_tr, _ = to_matrix(train_s)
        # fake snapshot-level fraud ratio as proxy label
        y_tr = torch.tensor([s.event_labels.float().mean()
                              for s in train_s], device=device)
        out  = model(x_tr.to(device))
        loss = nn.BCELoss()(out, y_tr)
        loss.backward(); opt.step()
    inf_time = time.perf_counter() - t0

    # Evaluate on test (event-level predictions broadcast from snapshot)
    model.eval()
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    all_true, all_score = [], []
    with torch.no_grad():
        x_te, te_labels = to_matrix(test_s)
        snap_scores = model(x_te.to(device)).cpu().numpy()
        for snap_score, lbl in zip(snap_scores, te_labels):
            all_true.extend(lbl.numpy().astype(int))
            all_score.extend([float(snap_score)] * len(lbl))
    y_true  = np.array(all_true)
    y_score = np.array(all_score)
    y_pred  = (y_score >= 0.5).astype(int)
    return dict(
        test_precision=float(precision_score(y_true, y_pred, zero_division=0)),
        test_recall=float(recall_score(y_true, y_pred, zero_division=0)),
        test_f1=float(f1_score(y_true, y_pred, zero_division=0)),
        test_roc_auc=float(roc_auc_score(y_true, y_score)
                           if len(np.unique(y_true)) > 1 else 0.),
        test_inference_time_s=inf_time,
    )


def run_isolation_forest(train_s, test_s, seed):
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import torch

    def snap_features(snaps):
        rows = []
        for s in snaps:
            row = torch.cat([s[nt].x.mean(0) for nt in s.node_types
                             if nt in s.node_types], dim=0)
            rows.append(row.numpy())
        return np.stack(rows)

    X_tr = snap_features(train_s)
    X_te = snap_features(test_s)
    clf  = IsolationForest(random_state=seed, contamination=0.10)
    clf.fit(X_tr)

    t0 = time.perf_counter()
    scores = -clf.score_samples(X_te)           # higher = more anomalous
    inf_t  = time.perf_counter() - t0
    snap_pred = (clf.predict(X_te) == -1).astype(int)

    # Broadcast to event level
    all_true, all_score, all_pred = [], [], []
    for i, s in enumerate(test_s):
        lbl = s.event_labels.numpy().astype(int)
        all_true.extend(lbl); all_score.extend([scores[i]] * len(lbl))
        all_pred.extend([snap_pred[i]] * len(lbl))
    y_true, y_score, y_pred = map(np.array, [all_true, all_score, all_pred])
    return dict(
        test_precision=float(precision_score(y_true, y_pred, zero_division=0)),
        test_recall=float(recall_score(y_true, y_pred, zero_division=0)),
        test_f1=float(f1_score(y_true, y_pred, zero_division=0)),
        test_roc_auc=float(roc_auc_score(y_true, y_score)
                           if len(np.unique(y_true)) > 1 else 0.),
        test_inference_time_s=inf_t,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42, 123, 2024, 3407, 7777])
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/synthetic/")
    parser.add_argument("--out-dir",  default="results/raw/baselines/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from data.temporal_dataset import TemporalCredentialDataset
    from graph.snapshots import chronological_split
    from training.train import build_model, train_one_epoch, compute_class_weight
    from training.evaluate import evaluate
    from training.early_stopping import EarlyStopping

    out_path = Path(args.out_dir); out_path.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        dataset   = TemporalCredentialDataset(args.data_dir, cfg["dataset"]["n_snapshots"])
        snapshots = list(dataset)
        input_dims = dataset.get_input_dims()
        train_ids, val_ids, test_ids = chronological_split(
            cfg["dataset"]["n_snapshots"], cfg["split"]["train_ratio"], cfg["split"]["val_ratio"])
        train_s = [snapshots[i] for i in train_ids]
        val_s   = [snapshots[i] for i in val_ids]
        test_s  = [snapshots[i] for i in test_ids]
        pos_w   = compute_class_weight(train_s, cfg)
        tcfg    = cfg["training"]

        # ── GNN models ────────────────────────────────────
        for mname in ["static_gcn", "tgat", "tgn", "tgnn"]:
            log.info("Baseline %s | seed=%d", mname, seed)
            model = build_model(mname, input_dims, cfg).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
            es    = EarlyStopping(patience=tcfg["early_stopping_patience"])
            for epoch in range(1, tcfg["epochs"] + 1):
                train_one_epoch(model, train_s, opt, device, pos_w)
                vm = evaluate(model, val_s, device)
                if es(1 - vm["f1"]): break
            tm = evaluate(model, test_s, device)
            result = dict(model=mname, seed=seed,
                          **{f"test_{k}": v for k, v in tm.items()})
            with open(out_path / f"{mname}_seed_{seed}.json", "w") as f:
                json.dump(result, f, indent=2)
            log.info("  %s seed=%d  F1=%.4f  AUC=%.4f", mname, seed, tm["f1"], tm["roc_auc"])

        # ── CNN ───────────────────────────────────────────
        log.info("Baseline CNN | seed=%d", seed)
        cnn_res = run_cnn_baseline(train_s, val_s, test_s, device, cfg, seed)
        with open(out_path / f"cnn_seed_{seed}.json", "w") as f:
            json.dump(dict(model="cnn", seed=seed, **cnn_res), f, indent=2)

        # ── Isolation Forest ──────────────────────────────
        log.info("Baseline IsolationForest | seed=%d", seed)
        iso_res = run_isolation_forest(train_s, test_s, seed)
        with open(out_path / f"isolation_forest_seed_{seed}.json", "w") as f:
            json.dump(dict(model="isolation_forest", seed=seed, **iso_res), f, indent=2)

    log.info("All baselines complete.")


if __name__ == "__main__":
    main()
