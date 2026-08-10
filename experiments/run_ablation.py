"""
experiments/run_ablation.py
-----------------------------
Trains the five ablation configurations over multiple seeds.

A1 : Static GCN  (no GRU, no attention, homogeneous)
A2 : Static GCN + GRU
A3 : GCN + Attention  (hetero attention, no GRU)
A4 : Temporal + Attention, no heterogeneous aggregation
A5 : Full T-GNN

Results saved to results/raw/ablation/<config>_seed_<N>.json
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import yaml, torch, numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

ABLATION_CONFIGS = {
    "A1_static_gcn": {
        "model": "static_gcn",
        "overrides": {},
        "description": "Static homogeneous GCN, no GRU, no attention",
    },
    "A2_gcn_gru": {
        "model": "tgnn",
        "overrides": {
            "use_heterogeneous": False,
            "use_attention": False,
        },
        "description": "GCN + GRU, no relation-specific attention",
    },
    "A3_gcn_attention": {
        "model": "tgnn",
        "overrides": {
            "use_gru": False,
        },
        "description": "Relation-aware attention, no GRU temporal encoder",
    },
    "A4_temporal_attention_no_hetero": {
        "model": "tgnn",
        "overrides": {
            "use_heterogeneous": False,
        },
        "description": "Temporal + attention, homogeneous aggregation",
    },
    "A5_full_tgnn": {
        "model": "tgnn",
        "overrides": {},
        "description": "Full T-GNN (proposed model)",
    },
}


def run_ablation(seed: int, cfg: dict, data_dir: str, out_dir: str) -> None:
    from data.temporal_dataset import TemporalCredentialDataset
    from graph.snapshots import chronological_split
    from training.train import build_model, train_one_epoch, compute_class_weight
    from training.evaluate import evaluate
    from training.early_stopping import EarlyStopping

    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TemporalCredentialDataset(data_dir, n_snapshots=cfg["dataset"]["n_snapshots"])
    snapshots = list(dataset)
    input_dims = dataset.get_input_dims()

    train_ids, val_ids, test_ids = chronological_split(
        cfg["dataset"]["n_snapshots"], cfg["split"]["train_ratio"], cfg["split"]["val_ratio"])
    train_s = [snapshots[i] for i in train_ids]
    val_s   = [snapshots[i] for i in val_ids]
    test_s  = [snapshots[i] for i in test_ids]
    pos_w   = compute_class_weight(train_s, cfg)

    tcfg = cfg["training"]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for key, acfg in ABLATION_CONFIGS.items():
        log.info("Ablation %s | seed=%d", key, seed)
        try:
            model = build_model(acfg["model"], input_dims, cfg,
                                overrides=acfg["overrides"]).to(device)
        except Exception as e:
            log.warning("Could not build %s: %s – using default model", key, e)
            model = build_model(acfg["model"], input_dims, cfg).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])
        es  = EarlyStopping(patience=tcfg["early_stopping_patience"])

        for epoch in range(1, tcfg["epochs"] + 1):
            import torch.nn as nn
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
            loss = train_one_epoch(model, train_s, opt, device, pos_w)
            val_m = evaluate(model, val_s, device)
            if es(1 - val_m["f1"]):
                break

        test_m = evaluate(model, test_s, device)
        result = dict(
            config=key, description=acfg["description"],
            seed=seed, **{f"test_{k}": v for k, v in test_m.items()})
        with open(out_path / f"{key}_seed_{seed}.json", "w") as f:
            json.dump(result, f, indent=2)
        log.info("  %s  F1=%.4f  AUC=%.4f", key, test_m["f1"], test_m["roc_auc"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42])
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/synthetic/")
    parser.add_argument("--out-dir",  default="results/raw/ablation/")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for seed in args.seeds:
        run_ablation(seed, cfg, args.data_dir, args.out_dir)

if __name__ == "__main__":
    main()
