"""
graph/snapshots.py
------------------
Assigns events to temporal snapshots and returns per-snapshot event DataFrames.

snapshot_id = floor(timestamp / snapshot_window_seconds)
Default window: 10 days  →  60 snapshots over 600 days.

NO future information crosses snapshot boundaries.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterator

SECONDS_PER_DAY = 86_400


def iter_snapshots(events: pd.DataFrame,
                   n_snapshots: int) -> Iterator[tuple[int, pd.DataFrame]]:
    """
    Yield (snapshot_id, snapshot_events) in chronological order.
    Each snapshot_events DataFrame contains only events in that window.
    """
    for snap_id in range(n_snapshots):
        snap_events = events[events["snapshot_id"] == snap_id]
        yield snap_id, snap_events


def get_cumulative_node_stats(events: pd.DataFrame,
                               up_to_snapshot: int) -> dict[str, pd.DataFrame]:
    """
    Compute node-level statistics from all events UP TO (and including)
    the given snapshot_id.  Used for leak-free feature computation.

    Returns dict with keys: institution_stats, student_stats,
    verifier_stats, credential_stats.
    """
    past = events[events["snapshot_id"] <= up_to_snapshot]

    inst_iss   = past[past["relation_type"] == "issues"].groupby("source_id").size().rename("issuance_count")
    inst_rev   = past[past["relation_type"] == "revokes"].groupby("source_id").size().rename("revocation_count")
    inst_coiss = past[past["relation_type"] == "co_issues"].groupby("source_id").size().rename("co_issuance_count")
    institution_stats = pd.concat([inst_iss, inst_rev, inst_coiss], axis=1).fillna(0).astype(int)

    stud_own = past[past["relation_type"] == "owns"].groupby("source_id").size().rename("credential_count")
    student_stats = stud_own.to_frame().fillna(0).astype(int)

    ver_ver  = past[past["relation_type"] == "verifies"].groupby("source_id").size().rename("verification_count")
    verifier_stats = ver_ver.to_frame().fillna(0).astype(int)

    cred_ver = past[past["relation_type"] == "verifies"].groupby("target_id").size().rename("times_verified")
    cred_mod = past[past["relation_type"] == "modifies"].groupby("target_id").size().rename("modifications")
    credential_stats = pd.concat([cred_ver, cred_mod], axis=1).fillna(0).astype(int)

    return dict(
        institution_stats=institution_stats,
        student_stats=student_stats,
        verifier_stats=verifier_stats,
        credential_stats=credential_stats,
    )


def chronological_split(n_snapshots: int,
                         train_ratio: float = 0.70,
                         val_ratio: float   = 0.15,
                         ) -> tuple[list[int], list[int], list[int]]:
    """
    Returns (train_ids, val_ids, test_ids) – contiguous snapshot ranges.

    For 60 snapshots (default):
        train : 0-41  (42 snapshots, 70%)
        val   : 42-50 (9  snapshots, 15%)
        test  : 51-59 (9  snapshots, 15%)
    """
    n_train = round(n_snapshots * train_ratio)
    n_val   = round(n_snapshots * val_ratio)
    n_test  = n_snapshots - n_train - n_val
    train_ids = list(range(0, n_train))
    val_ids   = list(range(n_train, n_train + n_val))
    test_ids  = list(range(n_train + n_val, n_snapshots))
    return train_ids, val_ids, test_ids
