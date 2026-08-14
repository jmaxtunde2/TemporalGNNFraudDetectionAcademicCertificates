"""
graph/heterogeneous_graph.py
-----------------------------
Builds a PyG HeteroData object for one temporal snapshot.

Node types  : student, institution, credential, verifier
Edge types  : (see graph/schema.py EDGE_TYPES)

Node features are computed from events UP TO the current snapshot
to prevent temporal leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from graph.schema import EDGE_TYPES
from graph.snapshots import get_cumulative_node_stats


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def build_node_features(
    students: pd.DataFrame,
    institutions: pd.DataFrame,
    verifiers: pd.DataFrame,
    credentials: pd.DataFrame,
    stats: dict[str, pd.DataFrame],
) -> dict[str, torch.Tensor]:
    """
    Build per-node-type feature tensors.
    All features are min-max normalised per snapshot.
    Only features observable at the snapshot boundary are included.
    """
    # ── Students ──────────────────────────────────────────
    # features: [type_id=0, credential_count]
    s_cred_cnt = (
        students["student_id"]
        .map(stats["student_stats"].get("credential_count", pd.Series(dtype=float)))
        .fillna(0).values
    )
    s_feats = np.stack([
        np.zeros(len(students)),          # type_id
        _minmax(s_cred_cnt),
    ], axis=1).astype(np.float32)

    # ── Institutions ──────────────────────────────────────
    # features: [type_id=1, accredited, is_fake, issuance_count,
    #            revocation_count, co_issuance_count]
    i_stats = stats["institution_stats"]
    i_iss = institutions["institution_id"].map(i_stats.get("issuance_count",   pd.Series(dtype=float))).fillna(0).values
    i_rev = institutions["institution_id"].map(i_stats.get("revocation_count", pd.Series(dtype=float))).fillna(0).values
    i_coi = institutions["institution_id"].map(i_stats.get("co_issuance_count",pd.Series(dtype=float))).fillna(0).values
    i_feats = np.stack([
        np.ones(len(institutions)),                      # type_id
        institutions["accredited"].values.astype(float),
        institutions["is_fake"].values.astype(float),
        _minmax(i_iss),
        _minmax(i_rev),
        _minmax(i_coi),
    ], axis=1).astype(np.float32)

    # ── Verifiers ─────────────────────────────────────────
    # features: [type_id=3, authorized, is_corrupt, verification_count]
    v_stats = stats["verifier_stats"]
    v_ver = verifiers["verifier_id"].map(v_stats.get("verification_count", pd.Series(dtype=float))).fillna(0).values
    v_feats = np.stack([
        np.full(len(verifiers), 3),
        verifiers["authorized"].values.astype(float),
        verifiers["is_corrupt"].values.astype(float),
        _minmax(v_ver),
    ], axis=1).astype(np.float32)

    # ── Credentials ───────────────────────────────────────
    # features: [type_id=2, status_is_revoked, times_verified, modifications]
    c_stats = stats["credential_stats"]
    c_ver = credentials["credential_id"].map(c_stats.get("times_verified", pd.Series(dtype=float))).fillna(0).values
    c_mod = credentials["credential_id"].map(c_stats.get("modifications",  pd.Series(dtype=float))).fillna(0).values
    c_rev = (credentials["status"] == "revoked").astype(float).values
    c_feats = np.stack([
        np.full(len(credentials), 2),
        c_rev,
        _minmax(c_ver),
        _minmax(c_mod),
    ], axis=1).astype(np.float32)

    return dict(
        student=torch.from_numpy(s_feats),
        institution=torch.from_numpy(i_feats),
        verifier=torch.from_numpy(v_feats),
        credential=torch.from_numpy(c_feats),
    )


def build_edge_index(snap_events: pd.DataFrame,
                     src_type: str, rel: str, tgt_type: str,
                     src_map: dict, tgt_map: dict) -> torch.Tensor | None:
    """Build a (2, E) edge_index tensor for one relation type."""
    mask = (
        snap_events["source_type"].eq(src_type) &
        snap_events["relation_type"].eq(rel)
    )
    if tgt_type != "institution":
        mask = mask & snap_events["target_type"].eq(tgt_type)
    else:
        mask = mask & snap_events["target_type"].eq(tgt_type)

    sub = snap_events[mask]
    if len(sub) == 0:
        return None

    src_ids = sub["source_id"].map(src_map).dropna().astype(int).values
    tgt_ids = sub["target_id"].map(tgt_map).dropna().astype(int).values
    # align lengths after potential dropna
    n = min(len(src_ids), len(tgt_ids))
    if n == 0:
        return None
    ei = torch.tensor(np.stack([src_ids[:n], tgt_ids[:n]]), dtype=torch.long)
    return ei


def build_hetero_graph(
    snapshot_id: int,
    all_events: pd.DataFrame,
    students: pd.DataFrame,
    institutions: pd.DataFrame,
    verifiers: pd.DataFrame,
    credentials: pd.DataFrame,
    snap_events: pd.DataFrame,
) -> HeteroData:
    """
    Build a HeteroData for one snapshot.

    Parameters
    ----------
    snapshot_id   : current snapshot index (used for cumulative stats)
    all_events    : full events DataFrame (for cumulative stats up to now)
    snap_events   : events that fall within this snapshot window
    """
    stats = get_cumulative_node_stats(all_events, snapshot_id)
    node_feats = build_node_features(students, institutions, verifiers, credentials, stats)

    # Index maps  entity_id → local tensor index
    stud_map  = {int(r): i for i, r in enumerate(students["student_id"])}
    inst_map  = {int(r): i for i, r in enumerate(institutions["institution_id"])}
    ver_map   = {int(r): i for i, r in enumerate(verifiers["verifier_id"])}
    cred_map  = {int(r): i for i, r in enumerate(credentials["credential_id"])}

    type_maps = {
        "student":     stud_map,
        "institution": inst_map,
        "verifier":    ver_map,
        "credential":  cred_map,
    }

    data = HeteroData()
    data["student"].x     = node_feats["student"]
    data["institution"].x = node_feats["institution"]
    data["verifier"].x    = node_feats["verifier"]
    data["credential"].x  = node_feats["credential"]

    # ── Edge types ────────────────────────────────────────
    for (src_type, rel, tgt_type) in EDGE_TYPES:
        ei = build_edge_index(snap_events, src_type, rel, tgt_type,
                              type_maps[src_type], type_maps[tgt_type])
        store = data[src_type, rel, tgt_type]
        if ei is not None and ei.shape[1] > 0:
            store.edge_index = ei
        else:
            store.edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ── Event-level targets ───────────────────────────────
    data.event_labels = torch.tensor(
        snap_events["fraud_label"].values, dtype=torch.float)
    data.event_src = torch.tensor(
        snap_events["source_id"].values, dtype=torch.long)
    data.event_tgt = torch.tensor(
        snap_events["target_id"].values, dtype=torch.long)
    data.event_src_type = snap_events["source_type"].values
    data.event_tgt_type = snap_events["target_type"].values
    data.event_relation  = torch.tensor(
        snap_events["relation_type"].map({
            "issues":0,"owns":1,"verifies":2,"co_issues":3,"revokes":4,"modifies":5
        }).fillna(0).astype(int).values, dtype=torch.long)
    data.event_timestamp = torch.tensor(
        snap_events["timestamp"].values, dtype=torch.float)
    if "on_chain" in snap_events.columns:
        data.event_on_chain = torch.tensor(
            snap_events["on_chain"].values, dtype=torch.float)
    else:
        # Default to zeros if missing (e.g. real data without this feature)
        data.event_on_chain = torch.zeros(len(snap_events), dtype=torch.float)
    data.snapshot_id = snapshot_id

    return data
