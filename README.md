# T-GNN: Temporal Heterogeneous GNN for Academic Credential Fraud Detection

> **Reviewer-revised implementation** — faithfully matches the methodology described in the paper.

## Data Notice

| | |
|---|---|
| 🔒 **Real Corilla data** | Private / unavailable in this repository |
| ✅ **Synthetic data** | Public / fully reproducible |

The default pipeline uses **only synthetic data**. Real Corilla data is loaded only when `--data-source real` is explicitly passed.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset (seed 42)
python data/generate_synthetic.py --seed 42

# 3. Validate dataset integrity
python data/validate_dataset.py

# 4. Train the T-GNN
python training/train.py --model tgnn --seed 42

# 5. Run all experiments end-to-end
python experiments/run_all.py --seeds 42 123 2024 3407 7777

# 6. Generate figures (after experiments)
python figures/figure5.py
python figures/figure6.py
python figures/figure7.py
```

## Architecture

```
Synthetic Dataset (2,000 students / 30 institutions / 200,000 events)
    ↓
Temporal Heterogeneous Graph (4 node types, 6 relation types, 60 snapshots)
    ↓
Relation-Aware GNN  (2 layers, per-relation W_r and attention α_r)
    ↓
GRU Temporal Encoder  (hidden dim = 128, across 60 snapshots)
    ↓
Event-Level Classifier  (concat src+tgt+rel_emb+time_enc → MLP → sigmoid)
    ↓
Binary Fraud Probability  ∈ (0, 1)
```

## Five Fraud Scenarios

| # | Scenario | Structural pattern |
|---|---|---|
| 1 | Fake Institution | Non-accredited institution with abnormally high issuance volume |
| 2 | Insider Corruption | Authorized verifier with burst verification activity |
| 3 | Identity Substitution | Multiple credentials for same student in short time window |
| 4 | Collusive Fraud | Reciprocal co-issuance between institution pairs |
| 5 | Retroactive Manipulation | Rapid revoke → re-issue sequences |

**Important**: these are five *generation scenarios*. The model output is **binary** (`fraud_label ∈ {0,1}`). `fraud_type` is metadata only.

## Chronological Split (60 snapshots, 10-day windows)

| Split | Snapshots | Days |
|---|---|---|
| Train | 0–41 | 0–419 |
| Val | 42–50 | 420–509 |
| Test | 51–59 | 510–599 |

## Project Structure

```
TGNN-code/
├── configs/default.yaml          # All hyperparameters
├── data/
│   ├── generate_synthetic.py     # Deterministic data generator
│   ├── validate_dataset.py       # Pre-training integrity checks
│   ├── temporal_dataset.py       # PyTorch Dataset wrapper
│   └── real_corilla_loader.py    # Private data interface (gated)
├── graph/
│   ├── schema.py                 # NodeType / RelationType enums
│   ├── relations.py              # Relation constants + embedding
│   ├── heterogeneous_graph.py    # HeteroData builder per snapshot
│   └── snapshots.py              # Snapshot utilities + split
├── models/
│   ├── relation_attention.py     # Relation-aware GNN layer
│   ├── temporal_gnn.py           # Full T-GNN (proposed)
│   ├── static_gcn.py             # Static GCN baseline
│   ├── tgat.py                   # TGAT baseline (Xu et al. 2020)
│   └── tgn.py                    # TGN baseline (Rossi et al. 2020)
├── training/
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Metrics computation
│   └── early_stopping.py        # EarlyStopping callback
├── experiments/
│   ├── run_baselines.py          # All baselines × 5 seeds
│   ├── run_ablation.py           # A1–A5 ablation study
│   ├── run_sensitivity.py        # Dim + window sweeps
│   └── run_all.py               # Full pipeline orchestrator
├── figures/
│   ├── figure5.py  figure6.py  figure7.py
├── results/raw/ aggregated/ logs/
├── notebooks/corilla_tgnn_colab.ipynb
├── REPRODUCIBILITY.md
├── IMPLEMENTATION_NOTES.md
└── CHANGELOG.md
```

## See Also

- `REPRODUCIBILITY.md` — exact commands, expected output files, hardware
- `IMPLEMENTATION_NOTES.md` — paper ambiguities and resolutions
- `CHANGELOG.md` — v1 vs v2 comparison

## Citation

If you use this code, cite the original paper (citation to be added) and note that the implementation corresponds to **v2.0** (reviewer-revised).
