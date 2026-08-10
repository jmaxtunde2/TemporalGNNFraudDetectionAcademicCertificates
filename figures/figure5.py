"""
figures/figure5.py  –  Baseline comparison bar chart (F1 and ROC-AUC).

Reads:  results/raw/baselines/*.json
Writes: figures/output/figure5_baselines.pdf  +  results/aggregated/baselines.csv

FAILS with an informative message if result files are absent.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")

RAW_DIR = Path("results/raw/baselines/")
OUT_DIR = Path("figures/output/"); OUT_DIR.mkdir(parents=True, exist_ok=True)
AGG_DIR = Path("results/aggregated/"); AGG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["static_gcn", "cnn", "isolation_forest", "tgat", "tgn", "tgnn"]
MODEL_LABELS = {
    "static_gcn": "Static GCN",
    "cnn": "CNN",
    "isolation_forest": "Iso. Forest",
    "tgat": "TGAT",
    "tgn": "TGN",
    "tgnn": "T-GNN (Ours)",
}

if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.json")):
    print(f"ERROR: No result files found in {RAW_DIR}.\n"
          "Run  python experiments/run_baselines.py  first.", file=sys.stderr)
    sys.exit(1)

records = []
for fp in RAW_DIR.glob("*.json"):
    with open(fp) as f:
        records.append(json.load(f))

df = pd.DataFrame(records)
agg = df.groupby("model").agg(
    f1_mean=("test_f1", "mean"), f1_std=("test_f1", "std"),
    auc_mean=("test_roc_auc", "mean"), auc_std=("test_roc_auc", "std"),
).reindex([m for m in MODEL_ORDER if m in df["model"].unique()])
agg.to_csv(AGG_DIR / "baselines.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(agg))
colors = plt.cm.Set2(np.linspace(0, 1, len(agg)))

for ax, metric, std_col, title in [
    (axes[0], "f1_mean", "f1_std", "F1-Score"),
    (axes[1], "auc_mean","auc_std","ROC-AUC"),
]:
    bars = ax.bar(x, agg[metric], yerr=agg[std_col], capsize=5,
                  color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in agg.index],
                       rotation=30, ha="right")
    ax.set_ylabel(title)
    ax.set_title(f"Baseline Comparison — {title}")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.4)
    for bar, val in zip(bars, agg[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out = OUT_DIR / "figure5_baselines.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
