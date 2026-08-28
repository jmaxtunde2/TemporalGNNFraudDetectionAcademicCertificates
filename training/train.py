"""
training/train.py
------------------
Main training script.

Usage:
    # Synthetic data (default, public, reproducible):
    python training/train.py --model tgnn --seed 42

    # Real Corilla data (private, requires CSVs):
    python training/train.py --model tgnn --seed 42 \\
        --data-source real --real-data-dir /path/to/csvs/

Chronological split (60 snapshots):
    Train : snapshots  0–41  (70%)
    Val   : snapshots 42–50  (15%)
    Test  : snapshots 51–59  (15%)

All results saved to results/raw/<model>/<seed>.json
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path

import torch
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SUPPORTED_MODELS = ["tgnn", "static_gcn", "tgat", "tgn"]


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, input_dims: dict, cfg: dict,
                overrides: dict | None = None):
    mcfg = cfg["model"]
    if overrides:
        mcfg = {**mcfg, **overrides}

    if model_name == "tgnn":
        from models.temporal_gnn import TemporalHeteroGNN
        return TemporalHeteroGNN(
            input_dims=input_dims,
            embedding_dim=mcfg["embedding_dim"],
            gru_hidden=mcfg["gru_hidden_dim"],
            n_layers=mcfg["n_gnn_layers"],
            dropout=mcfg["dropout"],
            time_enc=mcfg["time_encoding"],
            time_enc_dim=mcfg["time_encoding_dim"],
            use_heterogeneous=mcfg.get("use_heterogeneous", True),
            use_attention=mcfg.get("use_attention", True),
            use_gru=mcfg.get("use_gru", True),
        )
    if model_name == "static_gcn":
        from models.static_gcn import StaticGCN
        return StaticGCN(input_dims, hidden_dim=mcfg["embedding_dim"],
                         dropout=mcfg["dropout"])
    if model_name == "tgat":
        from models.tgat import TGAT
        return TGAT(input_dims, hidden_dim=mcfg["embedding_dim"],
                    dropout=mcfg["dropout"])
    if model_name == "tgn":
        from models.tgn import TGN
        return TGN(input_dims, hidden_dim=mcfg["embedding_dim"],
                   dropout=mcfg["dropout"])
    raise ValueError(f"Unknown model: {model_name}")


def compute_class_weight(snapshots, cfg) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from training snapshots."""
    all_labels = []
    for s in snapshots:
        all_labels.extend(s.event_labels.numpy().astype(int).tolist())
    n_neg = all_labels.count(0)
    n_pos = all_labels.count(1)
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos)


def collect_loss(model, snapshots, device, pos_weight):
    model.train()
    probs_list, labels_list = model(snapshots, device)
    loss = torch.tensor(0.0, device=device)
    n = 0
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device))
    for probs, labels in zip(probs_list, labels_list):
        if probs.numel() == 0:
            continue
        # use logits equivalent: re-compute from probs via inverse sigmoid
        logits = torch.log(probs / (1 - probs + 1e-8))
        loss  += criterion(logits, labels.to(device))
        n     += 1
    return loss / max(n, 1)


def train_one_epoch(model, snapshots, optimizer, device, pos_weight):
    loss = collect_loss(model, snapshots, device, pos_weight)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="tgnn", choices=SUPPORTED_MODELS)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--data-source",  default="synthetic",
                        choices=["synthetic", "real"])
    parser.add_argument("--data-dir",     default="data/synthetic/")
    parser.add_argument("--real-data-dir",default="data/real/")
    parser.add_argument("--output-dir",   default="results/raw/")
    parser.add_argument("--epochs",       type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("reproducibility", {}).get("save_environment", True):
        from reproducibility import capture_environment
        capture_environment(Path(args.output_dir).parent / "reproducibility" / "environment", Path(__file__).resolve().parents[1])
    log.info("Device: %s | Model: %s | Seed: %d", device, args.model, args.seed)

    # ── Load data ────────────────────────────────────────
    if args.data_source == "synthetic":
        from data.temporal_dataset import TemporalCredentialDataset
        dataset = TemporalCredentialDataset(
            args.data_dir, n_snapshots=cfg["dataset"]["n_snapshots"])
    else:
        from data.real_corilla_loader import load_real_corilla_data
        from data.generate_synthetic import generate
        log.warning("Real data source: events will be generated from Corilla entities.")
        entities = load_real_corilla_data(args.real_data_dir, cfg)
        # Events must be generated externally; only entities are loaded here.
        raise NotImplementedError(
            "Real-data event generation not yet implemented. "
            "Use --data-source synthetic."
        )

    snapshots = list(dataset)
    input_dims = dataset.get_input_dims()

    # ── Chronological split ───────────────────────────────
    from graph.snapshots import chronological_split
    train_ids, val_ids, test_ids = chronological_split(
        cfg["dataset"]["n_snapshots"],
        cfg["split"]["train_ratio"],
        cfg["split"]["val_ratio"],
    )
    log.info("Split: train=%s val=%s test=%s",
             [train_ids[0], train_ids[-1]],
             [val_ids[0], val_ids[-1]],
             [test_ids[0], test_ids[-1]])

    train_snaps = [snapshots[i] for i in train_ids]
    val_snaps   = [snapshots[i] for i in val_ids]
    test_snaps  = [snapshots[i] for i in test_ids]

    # ── Model ─────────────────────────────────────────────
    model = build_model(args.model, input_dims, cfg).to(device)
    log.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # ── Optimizer ─────────────────────────────────────────
    tcfg = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"])

    pos_weight = compute_class_weight(train_snaps, cfg)
    log.info("pos_weight (fraud / legit ratio): %.2f", pos_weight.item())

    # ── Early stopping ────────────────────────────────────
    from training.early_stopping import EarlyStopping
    from training.evaluate import evaluate
    early_stop = EarlyStopping(patience=tcfg["early_stopping_patience"])

    n_epochs = args.epochs or tcfg["epochs"]
    best_val_f1, best_state = 0.0, None
    history = []

    t_start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_snaps, optimizer,
                                     device, pos_weight)
        val_metrics = evaluate(model, val_snaps, device)
        val_loss    = collect_loss(model, val_snaps, device, pos_weight).item()

        row = dict(epoch=epoch, train_loss=train_loss,
                   val_loss=val_loss, **{f"val_{k}": v
                   for k, v in val_metrics.items()})
        history.append(row)

        log.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_f1=%.4f | val_auc=%.4f",
            epoch, n_epochs, train_loss, val_loss,
            val_metrics["f1"], val_metrics["roc_auc"],
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state  = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}

        if early_stop(val_loss):
            log.info("Early stopping at epoch %d", epoch)
            break

    train_time = time.perf_counter() - t_start

    # ── Test evaluation ───────────────────────────────────
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    prediction_dir = Path(args.output_dir).parent / "predictions" / args.model
    prediction_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = evaluate(
        model, test_snaps, device, return_predictions=True,
        prediction_output=str(prediction_dir / f"seed_{args.seed}_test.csv"),
        split_name="test",
    )
    log.info(
        "TEST  precision=%.4f  recall=%.4f  f1=%.4f  roc_auc=%.4f",
        test_metrics["precision"], test_metrics["recall"],
        test_metrics["f1"], test_metrics["roc_auc"],
    )

    # ── Save results ──────────────────────────────────────
    from reproducibility import environment_info, git_info, sha256_file
    config_hash = sha256_file(args.config) if Path(args.config).exists() else None
    result = dict(
        seed=args.seed, model=args.model,
        config=cfg["model"], config_file=args.config, config_sha256=config_hash,
        git=git_info(Path(__file__).resolve().parents[1]),
        environment=environment_info(),
        training_time_s=train_time,
        history=history,
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        test_f1=test_metrics["f1"],
        test_roc_auc=test_metrics["roc_auc"],
        test_inference_time_s=test_metrics["inference_time_s"],
        train_snapshot_range=[train_ids[0], train_ids[-1]],
        val_snapshot_range=[val_ids[0], val_ids[-1]],
        test_snapshot_range=[test_ids[0], test_ids[-1]],
    )

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Results saved → %s", out_path)

    # ── Save checkpoint ───────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),
               ckpt_dir / f"{args.model}_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
