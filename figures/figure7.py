"""
figures/figure7.py  –  Sensitivity analysis line plots.

Reads:  results/raw/sensitivity/*.json
Writes: figures/output/figure7_sensitivity.pdf  +  results/aggregated/sensitivity.csv

Two subplots:
  (a) F1 vs embedding dimension
  (b) F1 vs snapshot window (days)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt; import matplotlib; matplotlib.use("Agg")

RAW_DIR = Path("results/raw/sensitivity/")
OUT_DIR = Path("figures/output/"); OUT_DIR.mkdir(parents=True, exist_ok=True)
AGG_DIR = Path("results/aggregated/"); AGG_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.json")):
    print(f"ERROR: No sensitivity results in {RAW_DIR}.\n"
          "Run  python experiments/run_sensitivity.py  first.", file=sys.stderr)
    sys.exit(1)

records = []
for fp in RAW_DIR.glob("*.json"):
    with open(fp) as f:
        records.append(json.load(f))

df = pd.DataFrame(records)
df.to_csv(AGG_DIR / "sensitivity_raw.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, sweep, xlabel in [
    (axes[0], "embedding_dim",   "Embedding Dimension"),
    (axes[1], "snapshot_window", "Snapshot Window (days)"),
]:
    sub = df[df["sweep"] == sweep].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"No data for {sweep}", transform=ax.transAxes,
                ha="center"); continue
    agg = sub.groupby("value")["f1"].agg(["mean","std"]).reset_index()
    agg = agg.sort_values("value")
    ax.plot(agg["value"], agg["mean"], "o-", color="#1f77b4", linewidth=2,
            markersize=6, label="Mean F1")
    ax.fill_between(agg["value"],
                    agg["mean"] - agg["std"],
                    agg["mean"] + agg["std"],
                    alpha=0.2, color="#1f77b4", label="±1 std")
    ax.set_xlabel(xlabel); ax.set_ylabel("F1-Score")
    ax.set_title(f"Sensitivity — {xlabel}")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.4); ax.legend()

plt.tight_layout()
out = OUT_DIR / "figure7_sensitivity.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight"); print(f"Saved: {out}")
