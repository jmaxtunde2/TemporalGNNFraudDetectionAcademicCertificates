# T-GNN Reproducibility Package

This repository contains the source code, configuration, synthetic benchmark, experiment runners, and reproducibility instrumentation for the temporal heterogeneous GNN (T-GNN) academic-credential fraud experiments.

## Reproducibility principle

The repository records the actual configuration, seed, dataset hashes, environment, raw predictions, confusion matrix, ROC data, training history, timing, and peak GPU memory for every experiment. Reported manuscript values must be treated as historical reported results until they are reproduced from these raw outputs; raw values must never be fabricated to match a manuscript aggregate.

## Main configuration

- Synthetic entities: 2,000 students, 30 institutions, 150 verifiers, 20,000 credentials.
- Target event count: 200,000 (the generator may produce a small deviation because revocation events depend on generated credential status).
- Temporal snapshots: 60, with a 10-day window.
- Chronological split: train 0–41, validation 42–50, test 51–59.
- Model: 128-dimensional node embeddings, two relation-aware graph layers, GRU hidden size 128, sinusoidal time encoding dimension 64.
- Training: Adam, learning rate 0.001, weight decay 1e-5, 100 epochs maximum, early stopping patience 10, dropout 0.3.
- Main seeds: 42, 123, 2024, 3407, 7777.

## Commands

```bash
python -m reproducibility.capture_environment
python -m reproducibility.dataset_manifest
python -m reproducibility.experiment_manifest
python data/validate_dataset.py --data data/synthetic/ --config configs/default.yaml
python experiments/run_baselines.py --seeds 42 123 2024 3407 7777
python experiments/run_ablation.py --seeds 42 123 2024 3407 7777
python experiments/run_sensitivity.py --seeds 42 123
```

## Raw outputs

Baseline results are stored under `results/raw/baselines/` and predictions under `results/raw/predictions/<model>/`.

Ablation results are stored under `results/raw/ablation/` and predictions under `results/raw/predictions/<ablation>/`.

Each evaluation result records precision, recall, F1, ROC-AUC, inference time, throughput, peak GPU memory, sample count, confusion matrix, classification report, and the prediction-file path. Prediction CSV files contain the global snapshot identifier, ground-truth label, predicted probability, and thresholded prediction for every evaluated event.

## Ablation configurations

- A1: Static homogeneous GCN.
- A2: Homogeneous GCN + GRU.
- A3: Heterogeneous relation-aware attention without GRU.
- A4: Homogeneous graph attention + GRU.
- A5: Full heterogeneous relation-aware attention + GRU.

The ablation runner no longer silently falls back to the full T-GNN if an override is invalid. A failed configuration is an experiment failure and must be fixed rather than silently replaced.

## Environment capture

Run `python -m reproducibility.capture_environment` before a full experiment. The capture includes Python, OS, CPU, RAM where available, PyTorch, PyG, CUDA, GPU properties, package versions, `pip freeze`, `nvidia-smi`, and Git state.

## Dataset manifest

Run `python -m reproducibility.dataset_manifest` to create a machine-readable manifest containing row counts, column names, SHA-256 hashes of the five Parquet files, relation counts, fraud counts/types, per-snapshot counts, and the exact chronological split identifiers.

## Corilla real data

The repository contains the application-level loader/schema for private Corilla data. The distributed experiment pipeline uses the synthetic Parquet benchmark; private Corilla records must not be committed to this repository. Any manuscript claim about real-data training or blockchain extraction must be supported by a separate private data/extraction manifest.

## Important interpretation rule

This repository is being prepared to make every result traceable. If a newly reproduced result differs from a previously reported manuscript value, preserve the raw result and investigate the difference. Do not alter code, seeds, labels, or outputs solely to force agreement with a reported aggregate.
