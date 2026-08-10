# REPRODUCIBILITY.md

## Software Environment

| Package            | Version (minimum) |
|--------------------|-------------------|
| Python             | 3.10              |
| PyTorch            | 2.1.0             |
| PyTorch Geometric  | 2.4.0             |
| pandas             | 2.0.0             |
| numpy              | 1.24.0            |
| scikit-learn       | 1.3.0             |
| matplotlib         | 3.7.0             |
| PyYAML             | 6.0               |
| pyarrow            | 12.0.0            |

```bash
pip install -r requirements.txt
```

## Hardware

All experiments are runnable on CPU. GPU (CUDA 11.8+) strongly recommended for the 200,000-event dataset. Tested on NVIDIA A100 40 GB. Expected wall-clock time on CPU: ~4 h per seed for the full T-GNN.

## Random Seeds

```python
SEEDS = [42, 123, 2024, 3407, 7777]
```

## Step-by-Step Commands

### 1. Generate synthetic dataset (seed 42)
```bash
python data/generate_synthetic.py --seed 42 --config configs/default.yaml
```

### 2. Validate dataset
```bash
python data/validate_dataset.py --data data/synthetic/ --config configs/default.yaml
```

### 3. Train proposed T-GNN (one seed)
```bash
python training/train.py --model tgnn --seed 42
```

### 4. Train all baselines (all 5 seeds)
```bash
python experiments/run_baselines.py --seeds 42 123 2024 3407 7777
```

### 5. Run ablation study
```bash
python experiments/run_ablation.py --seeds 42 123 2024 3407 7777
```

### 6. Run sensitivity analysis
```bash
python experiments/run_sensitivity.py --seeds 42 123
```

### 7. Generate figures (requires results from steps 4–6)
```bash
python figures/figure5.py
python figures/figure6.py
python figures/figure7.py
```

### 8. Run everything end-to-end
```bash
python experiments/run_all.py --seeds 42 123 2024 3407 7777
```

## Configuration Files

- `configs/default.yaml` — master hyperparameter file

## Expected Output Files

```
data/synthetic/
    students.parquet
    institutions.parquet
    verifiers.parquet
    credentials.parquet
    events.parquet

results/raw/baselines/
    tgnn_seed_42.json ... tgnn_seed_7777.json
    static_gcn_seed_*.json
    tgat_seed_*.json
    tgn_seed_*.json
    cnn_seed_*.json
    isolation_forest_seed_*.json

results/raw/ablation/
    A1_static_gcn_seed_*.json  ...  A5_full_tgnn_seed_*.json

results/raw/sensitivity/
    emb_dim_*_seed_*.json
    snap_win_*_seed_*.json

results/aggregated/
    baselines.csv
    ablation.csv
    sensitivity_raw.csv

figures/output/
    figure5_baselines.pdf
    figure6_ablation.pdf
    figure7_sensitivity.pdf
```

## Data Privacy Notice

The synthetic dataset is fully self-contained and contains NO real Corilla customer data.

Real Corilla data (private) can be loaded via:
```bash
python training/train.py --data-source real --real-data-dir /path/to/private/
```

This requires the following files (not distributed):
- `students.csv`
- `institutions.csv`
- `certificates.csv`
- `verifiers.csv` (optional)

**NEVER commit these files to a public repository.**

## Chronological Split (60 snapshots)

| Split | Snapshot Range | Fraction |
|-------|---------------|----------|
| Train | 0 – 41        | 70%      |
| Val   | 42 – 50       | 15%      |
| Test  | 51 – 59       | 15%      |
