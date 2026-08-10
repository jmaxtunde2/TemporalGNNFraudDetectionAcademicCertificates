"""
figures/figure6.py  –  Ablation study bar chart.

Reads:  results/raw/ablation/*.json
Writes: figures/output/figure6_ablation.pdf  +  results/aggregated/ablation.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt; import matplotlib; matplotlib.use("Agg")

RAW_DIR = Path("results/raw/ablation/")
OUT_DIR = Path("figures/output/"); OUT_DIR.mkdir(parents=True, exist_ok=True)
AGG_DIR = Path("results/aggregated/"); AGG_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.json")):
    print(f"ERROR: No ablation results in {RAW_DIR}.\n"
          "Run  python experiments/run_ablation.py  first.", file=sys.stderr)
    sys.exit(1)

records = []
for fp in RAW_DIR.glob("*.json"):
    with open(fp) as f:
        records.append(json.load(f))

df = pd.DataFrame(records)
order = ["A1_static_gcn","A2_gcn_gru","A3_gcn_attention",
         "A4_temporal_attention_no_hetero","A5_full_tgnn"]
labels = ["A1\nStatic GCN","A2\nGCN+GRU","A3\nGCN+Attn",
          "A4\nTemporal+Attn\n(no hetero)","A5\nFull T-GNN"]

agg = df.groupby("config").agg(
    f1_mean=("test_f1","mean"), f1_std=("test_f1","std"),
    auc_mean=("test_roc_auc","mean"), auc_std=("test_roc_auc","std"),
).reindex([c for c in order if c in df["config"].unique()])
agg.to_csv(AGG_DIR / "ablation.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(agg))
colors = plt.cm.Paired(np.linspace(0, 1, len(agg)))

for ax, m, s, title in [
    (axes[0],"f1_mean","f1_std","F1-Score"),
    (axes[1],"auc_mean","auc_std","ROC-AUC"),
]:
    bars = ax.bar(x, agg[m], yerr=agg[s], capsize=5,
                  color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(agg)], fontsize=8)
    ax.set_ylabel(title); ax.set_ylim(0, 1.05)
    ax.set_title(f"Ablation Study — {title}"); ax.grid(axis="y", alpha=0.4)
    for bar, val in zip(bars, agg[m]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out = OUT_DIR / "figure6_ablation.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight"); print(f"Saved: {out}")
