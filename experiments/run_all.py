"""
experiments/run_all.py  –  Master orchestrator.

Runs: data generation → validation → baselines → ablation → sensitivity
"""
from __future__ import annotations
import argparse, logging, subprocess, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent


def run(cmd: list[str]) -> None:
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(BASE))
    if result.returncode != 0:
        log.error("FAILED: %s", " ".join(cmd))
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",  nargs="+", type=int, default=[42, 123, 2024, 3407, 7777])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed-for-data", type=int, default=42)
    args = parser.parse_args()
    seeds_str = [str(s) for s in args.seeds]

    log.info("═══ Phase 1: Generate synthetic data ═══")
    run([sys.executable, "data/generate_synthetic.py",
         "--seed", str(args.seed_for_data), "--config", args.config])

    log.info("═══ Phase 2: Validate dataset ═══")
    run([sys.executable, "data/validate_dataset.py", "--config", args.config])

    log.info("═══ Phase 3: Baselines ═══")
    run([sys.executable, "experiments/run_baselines.py",
         "--seeds", *seeds_str, "--config", args.config])

    log.info("═══ Phase 4: Ablation ═══")
    run([sys.executable, "experiments/run_ablation.py",
         "--seeds", *seeds_str, "--config", args.config])

    log.info("═══ Phase 5: Sensitivity ═══")
    run([sys.executable, "experiments/run_sensitivity.py",
         "--seeds", seeds_str[0], "--config", args.config])

    log.info("═══ Phase 6: Figures ═══")
    for fig in ["figures/figure5.py", "figures/figure6.py", "figures/figure7.py"]:
        run([sys.executable, fig])

    log.info("All phases complete.")


if __name__ == "__main__":
    main()
