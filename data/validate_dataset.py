"""
data/validate_dataset.py  –  Pre-training dataset validation.

Usage:
    python data/validate_dataset.py --data data/synthetic/
"""
from __future__ import annotations
import argparse, sys, logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

VALID_FRAUD_TYPES = {
    "none","fake_institution","insider_corruption",
    "identity_substitution","collusive_fraud","retroactive_manipulation"
}
VALID_RELATIONS = {"issues","owns","verifies","co_issues","revokes","modifies"}
VALID_SOURCE_TYPES = {"student","institution","credential","verifier"}

def check(condition: bool, msg: str) -> None:
    if not condition:
        log.error("FAIL  %s", msg)
        sys.exit(1)
    log.info("PASS  %s", msg)

def validate(data_dir: str, cfg_path: str = "configs/default.yaml") -> None:
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]

    p = Path(data_dir)
    students     = pd.read_parquet(p / "students.parquet")
    institutions = pd.read_parquet(p / "institutions.parquet")
    verifiers    = pd.read_parquet(p / "verifiers.parquet")
    credentials  = pd.read_parquet(p / "credentials.parquet")
    events       = pd.read_parquet(p / "events.parquet")

    # ── Entity counts ─────────────────────────────────────
    check(len(students)     == ds["n_students"],     f"Students == {ds['n_students']} (got {len(students)})")
    check(len(institutions) == ds["n_institutions"], f"Institutions == {ds['n_institutions']} (got {len(institutions)})")
    check(len(credentials)  == ds["n_credentials"],  f"Credentials == {ds['n_credentials']} (got {len(credentials)})")
    check(abs(len(events) - ds["n_events"]) < 1000, f"Events ~= {ds['n_events']} (got {len(events)})")
    n_snaps = events["snapshot_id"].nunique()
    check(n_snaps == ds["n_snapshots"], f"Snapshots == {ds['n_snapshots']} (got {n_snaps})")

    # ── Label validity ────────────────────────────────────
    check(events["fraud_label"].isin([0,1]).all(), "fraud_label ∈ {0,1}")
    check(events["fraud_type"].isin(VALID_FRAUD_TYPES).all(), "fraud_type values valid")
    check(events["relation_type"].isin(VALID_RELATIONS).all(), "relation_type values valid")
    check(events["source_type"].isin(VALID_SOURCE_TYPES).all(), "source_type values valid")
    
    # IN-01 validation
    check("on_chain" in events.columns, "on_chain feature column exists")
    check(events["on_chain"].isin([0,1]).all(), "on_chain ∈ {0,1}")

    # ── Timestamp ordering ────────────────────────────────
    check(events["timestamp"].is_monotonic_increasing, "Timestamps are sorted")

    # ── Endpoint existence ────────────────────────────────
    stud_ids  = set(students["student_id"])
    inst_ids  = set(institutions["institution_id"])
    cred_ids  = set(credentials["credential_id"])
    ver_ids   = set(verifiers["verifier_id"])
    src_stud  = events[events["source_type"]=="student"]["source_id"]
    src_inst  = events[events["source_type"]=="institution"]["source_id"]
    src_ver   = events[events["source_type"]=="verifier"]["source_id"]
    tgt_cred  = events[events["target_type"]=="credential"]["target_id"]
    check(src_stud.isin(stud_ids).all(), "All student source IDs exist")
    check(src_inst.isin(inst_ids).all(), "All institution source IDs exist")
    check(src_ver.isin(ver_ids).all(),   "All verifier source IDs exist")
    check(tgt_cred.isin(cred_ids).all(), "All credential target IDs exist")

    # ── No future leakage check (feature stats use only past events) ──
    # Validate that snapshot_id is computed correctly
    win_sec = ds["snapshot_window_days"] * 86400
    computed = (events["timestamp"] // win_sec).clip(0, ds["n_snapshots"]-1)
    check((computed == events["snapshot_id"]).all(), "snapshot_id consistent with timestamps")

    # ── Print summary ─────────────────────────────────────
    n_fraud = events["fraud_label"].sum()
    ftype   = events[events["fraud_label"]==1]["fraud_type"].value_counts()
    log.info(
        "\nDataset Summary\n---------------\n"
        "Students:      %d\nInstitutions:  %d\nVerifiers:     %d\n"
        "Credentials:   %d\nEvents:        %d\nSnapshots:     %d\n"
        "Fraud events:  %d (%.1f%%)\nLegit events:  %d\n"
        "\nFraud scenario distribution:\n%s",
        len(students), len(institutions), len(verifiers),
        len(credentials), len(events), n_snaps,
        n_fraud, 100*n_fraud/len(events), len(events)-n_fraud, ftype.to_string()
    )
    log.info("All validation checks PASSED.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/synthetic/")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    validate(args.data, args.config)

if __name__ == "__main__":
    main()
