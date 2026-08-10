"""
experiments/run_sensitivity.py
-------------------------------
Automated sensitivity experiments.

Sweeps:
  1. Embedding dimension: [16, 32, 64, 128, 192, 256, 512]
  2. Snapshot window (days): [1, 5, 10, 15, 20, 30, 45, 60]

Every point is from an actual experiment (no smoothing / fabrication).
Results saved to results/raw/sensitivity/
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np, yaml, torch

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_one(seed, cfg, data_dir, emb_dim=None, snap_window=None):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Apply overrides to config
    run_cfg = yaml.safe_load(yaml.dump(cfg))   # deep copy
    if emb_dim is not None:
        run_cfg["model"]["embedding_dim"] = emb_dim
        run_cfg["model"]["gru_hidden_dim"] = emb_dim
    if snap_window is not None:
        run_cfg["dataset"]["snapshot_window_days"] = snap_window
        n_snaps = run_cfg["dataset"]["total_days"] // snap_window
        run_cfg["dataset"]["n_snapshots"] = n_snaps

    from data.temporal_dataset import TemporalCredentialDataset
    from graph.snapshots import chronological_split
    from training.train import build_model, train_one_epoch, compute_class_weight
    from training.evaluate import evaluate
    from training.early_stopping import EarlyStopping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TemporalCredentialDataset(data_dir, run_cfg["dataset"]["n_snapshots"])
    snapshots = list(dataset)
    input_dims = dataset.get_input_dims()

    train_ids, val_ids, test_ids = chronological_split(
        run_cfg["dataset"]["n_snapshots"],
        run_cfg["split"]["train_ratio"], run_cfg["split"]["val_ratio"])
    train_s = [snapshots[i] for i in train_ids]
    val_s   = [snapshots[i] for i in val_ids]
    test_s  = [snapshots[i] for i in test_ids]
    pos_w   = compute_class_weight(train_s, run_cfg)

    model = build_model("tgnn", input_dims, run_cfg).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=run_cfg["training"]["learning_rate"])
    es    = EarlyStopping(patience=run_cfg["training"]["early_stopping_patience"])

    for epoch in range(1, run_cfg["training"]["epochs"] + 1):
        train_one_epoch(model, train_s, opt, device, pos_w)
        vm = evaluate(model, val_s, device)
        if es(1 - vm["f1"]): break

    return evaluate(model, test_s, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42, 123])
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/synthetic/")
    parser.add_argument("--out-dir",  default="results/raw/sensitivity/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_path = Path(args.out_dir); out_path.mkdir(parents=True, exist_ok=True)

    # ── Embedding dim sweep ───────────────────────────────
    for dim in cfg["sensitivity"]["embedding_dims"]:
        for seed in args.seeds:
            log.info("Sensitivity emb_dim=%d seed=%d", dim, seed)
            metrics = run_one(seed, cfg, args.data_dir, emb_dim=dim)
            result  = dict(sweep="embedding_dim", value=dim, seed=seed, **metrics)
            fname   = out_path / f"emb_dim_{dim}_seed_{seed}.json"
            with open(fname, "w") as f:
                json.dump(result, f, indent=2)
            log.info("  dim=%d seed=%d F1=%.4f", dim, seed, metrics["f1"])

    # ── Snapshot window sweep ─────────────────────────────
    for win in cfg["sensitivity"]["snapshot_windows"]:
        for seed in args.seeds:
            log.info("Sensitivity snap_win=%d seed=%d", win, seed)
            # Re-generate data with new window
            from data.generate_synthetic import generate
            tmp_dir = f"/tmp/sensitivity_win{win}/"
            snap_cfg = yaml.safe_load(yaml.dump(cfg))
            snap_cfg["dataset"]["snapshot_window_days"] = win
            snap_cfg["dataset"]["n_snapshots"] = cfg["dataset"]["total_days"] // win
            generate(seed, snap_cfg, tmp_dir)
            metrics = run_one(seed, snap_cfg, tmp_dir, snap_window=win)
            result  = dict(sweep="snapshot_window", value=win, seed=seed, **metrics)
            fname   = out_path / f"snap_win_{win}_seed_{seed}.json"
            with open(fname, "w") as f:
                json.dump(result, f, indent=2)
            log.info("  win=%d seed=%d F1=%.4f", win, seed, metrics["f1"])

    log.info("Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
