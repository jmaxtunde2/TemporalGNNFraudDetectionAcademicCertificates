# CHANGELOG.md

## v2.0.0 — Paper-faithful refactor (this release)

### What changed

| Component | v1.0 (original) | v2.0 (this release) |
|---|---|---|
| Entry point | `tgnnAcadFraud.py` (230 lines) | 25+ modules, proper package structure |
| Data | Loads private Corilla CSVs directly | Synthetic-only default; real data isolated in `data/real_corilla_loader.py` |
| Data scale | Varies with CSV size | 2,000 students / 30 institutions / 20,000 credentials / 200,000 events / 60 snapshots |
| Augmentation | Random, unseeded `np.random` | Deterministic, 5 seeded fraud generators |
| Fraud labels | Node-level `revoked ∈ {0,1}` randomly from CSV | Event-level `fraud_label ∈ {0,1}` derived from structural/temporal patterns |
| Fraud scenarios | None | 5 explicit generators (fake institution, insider corruption, identity substitution, collusion, retroactive manipulation) |
| Graph type | Homogeneous `networkx.Graph` | Heterogeneous `HeteroData` with 4 node types, 6 edge types |
| GNN | 2-layer `GCNConv(3,16,8)` | 2-layer `RelationAwareConv` with per-relation W_r and alpha_r |
| Temporal | None | GRU encoder across 60 snapshots |
| Classifier | Node-level `Linear(8,2)` + cross-entropy | Event-level `concat(src,tgt,rel,time_enc) → MLP → sigmoid` |
| Time encoding | None | Sinusoidal (TGAT-style) or scalar delta |
| Data split | Random 80/20 | Chronological 70/15/15 (snapshot ranges: 0-41 / 42-50 / 51-59) |
| Loss function | `CrossEntropyLoss` (multi-class) | Weighted `BCEWithLogitsLoss` (binary, class-balanced) |
| Baselines | None | CNN, Isolation Forest, Static GCN, TGAT, TGN |
| Ablation | None | A1–A5 (5 configurations) |
| Sensitivity | None | Embedding dim sweep + snapshot window sweep |
| Seeds | None | 5 seeds: [42, 123, 2024, 3407, 7777] |
| Results | Printed to stdout | JSON per run + aggregated CSV |
| Figures | None | 3 scripts reading raw JSON, failing if missing |
| Documentation | None | README, REPRODUCIBILITY.md, IMPLEMENTATION_NOTES.md |
| Config | Hard-coded in script | `configs/default.yaml` |
| Private data | Loaded unconditionally | Gated behind `--data-source real` |

### Original file preserved

`tgnnAcadFraud_ORIGINAL.py` — the original 230-line prototype is preserved verbatim.

### Result comparison

> **Note**: New results from the corrected implementation will differ from those
> reported in the v1 manuscript. This is expected and scientifically correct.
> The manuscript should be updated based on new experimental output.
>
> A comparison table will be added here after experiments complete:
>
> | Metric | Manuscript (v1) | Reproducible (v2) | Δ |
> |--------|-----------------|-------------------|---|
> | F1     | —               | (run experiments) | — |
> | AUC    | —               | (run experiments) | — |

### Known issues in v1

1. Real Corilla CSV paths hard-coded — fails on any machine without the files.
2. Random seed not set → non-reproducible runs.
3. `revoked` column used as fraud label — scientifically incorrect.
4. Static GCN presented as "temporal GNN" in manuscript.
5. Homogeneous GCN presented as "heterogeneous" in manuscript.
6. 5 output classes not implemented despite paper describing binary classification.
7. Random 80/20 split causes temporal leakage.
8. No baselines, no ablation, no sensitivity analysis.
