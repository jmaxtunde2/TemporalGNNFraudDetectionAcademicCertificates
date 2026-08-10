"""
data/generate_synthetic.py  –  Deterministic synthetic dataset generator.

Usage:
    python data/generate_synthetic.py --seed 42
    python data/generate_synthetic.py --seed 42 --config configs/default.yaml --output data/synthetic/

Produces (in output dir):
    students.parquet, institutions.parquet, verifiers.parquet,
    credentials.parquet, events.parquet

Event counts (default config, total = 200,000):
    issuance     20,000   institution → credential
    ownership    20,000   student     → credential
    verification 120,000  verifier    → credential
    revocation   10,000   institution → credential
    modification 10,000   institution → credential
    co_issuance  20,000   institution → institution
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────
SECONDS_PER_DAY = 86_400
FRAUD_TYPES = [
    "none",
    "fake_institution",
    "insider_corruption",
    "identity_substitution",
    "collusive_fraud",
    "retroactive_manipulation",
]


# ════════════════════════════════════════════════════════════
# Entity generators
# ════════════════════════════════════════════════════════════

def generate_students(rng: np.random.Generator, n: int, total_days: int) -> pd.DataFrame:
    reg_times = rng.integers(0, total_days * SECONDS_PER_DAY, size=n)
    return pd.DataFrame({
        "student_id": np.arange(n),
        "registration_time": np.sort(reg_times),
    })


def generate_institutions(rng: np.random.Generator, n: int, cfg_fraud: dict) -> pd.DataFrame:
    n_fake = cfg_fraud["fake_institution"]["n_fake_institutions"]
    accredited = np.ones(n, dtype=int)
    fake_ids = rng.choice(n, size=min(n_fake, n), replace=False)
    accredited[fake_ids] = 0
    inst_types = rng.choice(["university", "polytechnic", "college", "professional"], size=n)
    return pd.DataFrame({
        "institution_id": np.arange(n),
        "accredited": accredited,
        "institution_type": inst_types,
        "is_fake": np.isin(np.arange(n), fake_ids).astype(int),
    })


def generate_verifiers(rng: np.random.Generator,
                       institutions: pd.DataFrame,
                       n_per_institution: int,
                       cfg_fraud: dict) -> pd.DataFrame:
    n_corrupt = cfg_fraud["insider_corruption"]["n_corrupt_verifiers"]
    rows = []
    v_id = 0
    for inst_id in institutions["institution_id"]:
        for _ in range(n_per_institution):
            rows.append({"verifier_id": v_id, "institution_id": inst_id, "authorized": 1})
            v_id += 1
    df = pd.DataFrame(rows)
    corrupt_ids = rng.choice(len(df), size=min(n_corrupt, len(df)), replace=False)
    df["is_corrupt"] = 0
    df.loc[corrupt_ids, "is_corrupt"] = 1
    return df


def generate_credentials(rng: np.random.Generator,
                         n: int,
                         students: pd.DataFrame,
                         institutions: pd.DataFrame,
                         total_days: int) -> pd.DataFrame:
    n_students = len(students)
    n_insts = len(institutions)
    issue_times = np.sort(rng.integers(0, total_days * SECONDS_PER_DAY, size=n))
    student_ids = rng.integers(0, n_students, size=n)
    inst_ids = rng.integers(0, n_insts, size=n)
    # 50% revoked to provide rich temporal signal
    revoke_mask = rng.random(n) < 0.50
    revoke_times = np.where(
        revoke_mask,
        issue_times + rng.integers(1 * SECONDS_PER_DAY, 200 * SECONDS_PER_DAY, size=n),
        -1,
    )
    status = np.where(revoke_mask, "revoked", "active")
    return pd.DataFrame({
        "credential_id": np.arange(n),
        "student_id": student_ids,
        "institution_id": inst_ids,
        "issue_time": issue_times,
        "revocation_time": revoke_times,
        "status": status,
    })


# ════════════════════════════════════════════════════════════
# Event generators  (each relation type)
# ════════════════════════════════════════════════════════════

def _event_rows(event_ids, timestamps, src_ids, src_types, tgt_ids, tgt_types,
                rel_types, cred_ids) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": event_ids,
        "timestamp": timestamps,
        "source_id": src_ids,
        "source_type": src_types,
        "target_id": tgt_ids,
        "target_type": tgt_types,
        "relation_type": rel_types,
        "credential_id": cred_ids,
        "fraud_label": 0,
        "fraud_type": "none",
    })


def gen_issuance_events(credentials: pd.DataFrame, start_eid: int) -> pd.DataFrame:
    n = len(credentials)
    return _event_rows(
        np.arange(start_eid, start_eid + n),
        credentials["issue_time"].values,
        credentials["institution_id"].values, "institution",
        credentials["credential_id"].values,  "credential",
        "issues",
        credentials["credential_id"].values,
    )


def gen_ownership_events(credentials: pd.DataFrame, start_eid: int) -> pd.DataFrame:
    n = len(credentials)
    ts = credentials["issue_time"].values + SECONDS_PER_DAY
    return _event_rows(
        np.arange(start_eid, start_eid + n),
        ts,
        credentials["student_id"].values, "student",
        credentials["credential_id"].values, "credential",
        "owns",
        credentials["credential_id"].values,
    )


def gen_verification_events(rng: np.random.Generator,
                             credentials: pd.DataFrame,
                             verifiers: pd.DataFrame,
                             n_target: int,
                             start_eid: int,
                             total_days: int) -> pd.DataFrame:
    n_creds = len(credentials)
    n_vers = len(verifiers)
    cred_ids = rng.integers(0, n_creds, size=n_target)
    ver_ids = rng.integers(0, n_vers, size=n_target)
    issue_ts = credentials["issue_time"].values[cred_ids]
    max_ts = total_days * SECONDS_PER_DAY
    offsets = rng.integers(1 * SECONDS_PER_DAY, 300 * SECONDS_PER_DAY, size=n_target)
    ts = np.clip(issue_ts + offsets, 0, max_ts - 1)
    return _event_rows(
        np.arange(start_eid, start_eid + n_target),
        ts,
        ver_ids, "verifier",
        cred_ids, "credential",
        "verifies",
        cred_ids,
    )


def gen_revocation_events(rng: np.random.Generator,
                           credentials: pd.DataFrame,
                           n_target: int,
                           start_eid: int) -> pd.DataFrame:
    revoked = credentials[credentials["status"] == "revoked"]
    if len(revoked) == 0:
        revoked = credentials
    idx = rng.choice(len(revoked), size=min(n_target, len(revoked)), replace=False)
    sel = revoked.iloc[idx]
    n = len(sel)
    ts = sel["revocation_time"].values.astype(np.int64)
    ts = np.where(ts < 0, sel["issue_time"].values + 10 * SECONDS_PER_DAY, ts)
    return _event_rows(
        np.arange(start_eid, start_eid + n),
        ts,
        sel["institution_id"].values, "institution",
        sel["credential_id"].values,  "credential",
        "revokes",
        sel["credential_id"].values,
    )


def gen_modification_events(rng: np.random.Generator,
                             credentials: pd.DataFrame,
                             n_target: int,
                             start_eid: int,
                             total_days: int) -> pd.DataFrame:
    idx = rng.choice(len(credentials), size=min(n_target, len(credentials)), replace=False)
    sel = credentials.iloc[idx]
    n = len(sel)
    offsets = rng.integers(5 * SECONDS_PER_DAY, 100 * SECONDS_PER_DAY, size=n)
    ts = np.clip(sel["issue_time"].values + offsets, 0, total_days * SECONDS_PER_DAY - 1)
    return _event_rows(
        np.arange(start_eid, start_eid + n),
        ts,
        sel["institution_id"].values, "institution",
        sel["credential_id"].values,  "credential",
        "modifies",
        sel["credential_id"].values,
    )


def gen_co_issuance_events(rng: np.random.Generator,
                            credentials: pd.DataFrame,
                            n_institutions: int,
                            n_target: int,
                            start_eid: int,
                            total_days: int) -> pd.DataFrame:
    src_insts = rng.integers(0, n_institutions, size=n_target)
    tgt_insts = rng.integers(0, n_institutions, size=n_target)
    same = src_insts == tgt_insts
    tgt_insts[same] = (tgt_insts[same] + 1) % n_institutions
    cred_ids = rng.integers(0, len(credentials), size=n_target)
    ts = rng.integers(0, total_days * SECONDS_PER_DAY, size=n_target)
    return _event_rows(
        np.arange(start_eid, start_eid + n_target),
        ts,
        src_insts, "institution",
        tgt_insts, "institution",
        "co_issues",
        cred_ids,
    )


# ════════════════════════════════════════════════════════════
# Fraud label assignment
# ════════════════════════════════════════════════════════════

def assign_fake_institution_fraud(events: pd.DataFrame,
                                   institutions: pd.DataFrame,
                                   cfg: dict) -> pd.DataFrame:
    """Mark issuance events from non-accredited institutions with
    abnormally high issuance volume."""
    fake_ids = set(institutions[institutions["is_fake"] == 1]["institution_id"])
    if not fake_ids:
        return events
    iss = events[events["relation_type"] == "issues"]
    counts = iss.groupby("source_id").size()
    threshold = counts.mean() * cfg["high_issuance_multiplier"]
    high_vol = set(counts[counts > threshold].index)
    suspicious = fake_ids & high_vol
    mask = (
        events["source_type"].eq("institution") &
        events["source_id"].isin(suspicious)
    )
    events.loc[mask, "fraud_label"] = 1
    events.loc[mask, "fraud_type"] = "fake_institution"
    return events


def assign_insider_corruption_fraud(events: pd.DataFrame,
                                     verifiers: pd.DataFrame,
                                     cfg: dict) -> pd.DataFrame:
    """Mark verification events during burst windows by corrupt verifiers."""
    corrupt_ids = set(verifiers[verifiers["is_corrupt"] == 1]["verifier_id"])
    if not corrupt_ids:
        return events
    burst_sec = cfg["burst_window_days"] * SECONDS_PER_DAY
    threshold = cfg["burst_threshold"]
    ver_evts = events[
        events["relation_type"].eq("verifies") &
        events["source_type"].eq("verifier") &
        events["source_id"].isin(corrupt_ids)
    ].copy()
    burst_eids: set = set()
    for v_id in corrupt_ids:
        v = ver_evts[ver_evts["source_id"] == v_id].sort_values("timestamp")
        if len(v) < threshold:
            continue
        ts = v["timestamp"].values
        for i in range(len(ts)):
            window = (ts >= ts[i]) & (ts <= ts[i] + burst_sec)
            if window.sum() >= threshold:
                burst_eids.update(v[window]["event_id"].values)
    events.loc[events["event_id"].isin(burst_eids), "fraud_label"] = 1
    events.loc[events["event_id"].isin(burst_eids), "fraud_type"] = "insider_corruption"
    return events


def assign_identity_substitution_fraud(events: pd.DataFrame,
                                        credentials: pd.DataFrame,
                                        cfg: dict) -> pd.DataFrame:
    """Mark issuance/ownership events where a student receives multiple
    credentials in an implausibly short window."""
    win_sec = cfg["window_days"] * SECONDS_PER_DAY
    min_creds = cfg["min_credentials_in_window"]
    sub_eids: set = set()
    for stud_id in credentials["student_id"].unique():
        sc = credentials[credentials["student_id"] == stud_id].sort_values("issue_time")
        if len(sc) < min_creds:
            continue
        ts = sc["issue_time"].values
        for i in range(len(ts)):
            window = (ts >= ts[i]) & (ts <= ts[i] + win_sec)
            if window.sum() >= min_creds:
                cred_ids_w = sc[window]["credential_id"].values
                mask = events["credential_id"].isin(cred_ids_w) & \
                       events["relation_type"].isin(["issues", "owns"])
                sub_eids.update(events[mask]["event_id"].values)
    events.loc[events["event_id"].isin(sub_eids), "fraud_label"] = 1
    events.loc[events["event_id"].isin(sub_eids), "fraud_type"] = "identity_substitution"
    return events


def assign_collusive_fraud(events: pd.DataFrame,
                            cfg: dict,
                            rng: np.random.Generator,
                            n_institutions: int) -> pd.DataFrame:
    """Mark co-issuance events involving designated colluding pairs."""
    n_pairs = cfg["n_colluding_pairs"]
    min_interact = cfg["min_reciprocal_interactions"]
    inst_ids = np.arange(n_institutions)
    pairs: list[tuple[int, int]] = []
    chosen = rng.choice(n_institutions, size=min(n_pairs * 2, n_institutions), replace=False)
    for i in range(0, len(chosen) - 1, 2):
        pairs.append((int(chosen[i]), int(chosen[i + 1])))
    col_eids: set = set()
    coiss = events[events["relation_type"] == "co_issues"]
    for a, b in pairs:
        ab = coiss[coiss["source_id"].eq(a) & coiss["target_id"].eq(b)]
        ba = coiss[coiss["source_id"].eq(b) & coiss["target_id"].eq(a)]
        if len(ab) + len(ba) >= min_interact:
            col_eids.update(ab["event_id"].values)
            col_eids.update(ba["event_id"].values)
    events.loc[events["event_id"].isin(col_eids), "fraud_label"] = 1
    events.loc[events["event_id"].isin(col_eids), "fraud_type"] = "collusive_fraud"
    return events


def assign_retroactive_manipulation_fraud(events: pd.DataFrame,
                                           credentials: pd.DataFrame,
                                           cfg: dict) -> pd.DataFrame:
    """Mark revoke + re-issue sequences where revocation and reissuance
    occur within a short window."""
    win_sec = cfg["revoke_reissue_window_days"] * SECONDS_PER_DAY
    revoked = credentials[credentials["status"] == "revoked"].copy()
    retro_eids: set = set()
    rev_evts = events[events["relation_type"] == "revokes"]
    for _, row in revoked.iterrows():
        cid = row["credential_id"]
        issue_t = row["issue_time"]
        rev_t = row["revocation_time"]
        if rev_t < 0:
            continue
        # look for any re-issuance within window after revocation
        reissued = events[
            events["relation_type"].eq("issues") &
            events["credential_id"].ne(cid) &
            events["source_id"].eq(row["institution_id"]) &
            events["timestamp"].between(rev_t, rev_t + win_sec)
        ]
        if len(reissued) > 0:
            # flag the revocation event and re-issuance events
            retro_eids.update(rev_evts[rev_evts["credential_id"] == cid]["event_id"].values)
            retro_eids.update(reissued["event_id"].values)
    events.loc[events["event_id"].isin(retro_eids), "fraud_label"] = 1
    events.loc[events["event_id"].isin(retro_eids), "fraud_type"] = "retroactive_manipulation"
    return events


# ════════════════════════════════════════════════════════════
# Main generator
# ════════════════════════════════════════════════════════════

def generate(seed: int, cfg: dict, output_dir: str) -> None:
    rng = np.random.default_rng(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ds = cfg["dataset"]
    n_students = ds["n_students"]
    n_insts = ds["n_institutions"]
    n_vpinst = ds["n_verifiers_per_institution"]
    n_creds = ds["n_credentials"]
    n_events_target = ds["n_events"]
    total_days = ds["total_days"]
    fraud_cfg = cfg["fraud"]

    log.info("Seed=%d | Generating entities …", seed)
    students = generate_students(rng, n_students, total_days)
    institutions = generate_institutions(rng, n_insts, fraud_cfg)
    verifiers = generate_verifiers(rng, institutions, n_vpinst, fraud_cfg)
    credentials = generate_credentials(rng, n_creds, students, institutions, total_days)

    log.info("Generating events (target=%d) …", n_events_target)
    # Fixed distribution summing to n_events_target
    n_iss   = n_creds           # 20,000
    n_own   = n_creds           # 20,000
    n_ver   = 120_000
    n_rev   = 10_000
    n_mod   = 10_000
    n_coiss = n_events_target - n_iss - n_own - n_ver - n_rev - n_mod  # 20,000

    eid = 0
    iss   = gen_issuance_events(credentials, eid);  eid += len(iss)
    own   = gen_ownership_events(credentials, eid); eid += len(own)
    ver   = gen_verification_events(rng, credentials, verifiers, n_ver, eid, total_days); eid += len(ver)
    rev   = gen_revocation_events(rng, credentials, n_rev, eid); eid += len(rev)
    mod   = gen_modification_events(rng, credentials, n_mod, eid, total_days); eid += len(mod)
    coiss = gen_co_issuance_events(rng, credentials, n_insts, n_coiss, eid, total_days)

    events = pd.concat([iss, own, ver, rev, mod, coiss], ignore_index=True)
    events = events.sort_values("timestamp").reset_index(drop=True)
    events["event_id"] = np.arange(len(events))

    log.info("Assigning fraud labels …")
    events = assign_fake_institution_fraud(events, institutions, fraud_cfg["fake_institution"])
    events = assign_insider_corruption_fraud(events, verifiers, fraud_cfg["insider_corruption"])
    events = assign_identity_substitution_fraud(events, credentials, fraud_cfg["identity_substitution"])
    events = assign_collusive_fraud(events, fraud_cfg["collusive_fraud"], rng, n_insts)
    events = assign_retroactive_manipulation_fraud(events, credentials, fraud_cfg["retroactive_manipulation"])

    # Snapshot assignment
    win_sec = ds["snapshot_window_days"] * SECONDS_PER_DAY
    events["snapshot_id"] = (events["timestamp"] // win_sec).astype(int)
    events["snapshot_id"] = events["snapshot_id"].clip(0, ds["n_snapshots"] - 1)

    # ── Save ──────────────────────────────────────────────
    students.to_parquet(f"{output_dir}/students.parquet", index=False)
    institutions.to_parquet(f"{output_dir}/institutions.parquet", index=False)
    verifiers.to_parquet(f"{output_dir}/verifiers.parquet", index=False)
    credentials.to_parquet(f"{output_dir}/credentials.parquet", index=False)
    events.to_parquet(f"{output_dir}/events.parquet", index=False)

    # ── Summary ───────────────────────────────────────────
    fraud_counts = events[events["fraud_label"] == 1]["fraud_type"].value_counts()
    n_fraud = events["fraud_label"].sum()
    log.info(
        "\nDataset Summary\n"
        "---------------\n"
        "Students:      %d\n"
        "Institutions:  %d\n"
        "Verifiers:     %d\n"
        "Credentials:   %d\n"
        "Events:        %d\n"
        "Snapshots:     %d\n"
        "Fraud events:  %d (%.1f%%)\n"
        "Legit events:  %d\n"
        "\nFraud scenario distribution:\n%s",
        len(students), len(institutions), len(verifiers),
        len(credentials), len(events),
        events["snapshot_id"].nunique(),
        n_fraud, 100 * n_fraud / len(events),
        len(events) - n_fraud,
        fraud_counts.to_string(),
    )
    log.info("Saved to %s", output_dir)


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic T-GNN dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/synthetic/")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    generate(args.seed, cfg, args.output)


if __name__ == "__main__":
    main()
