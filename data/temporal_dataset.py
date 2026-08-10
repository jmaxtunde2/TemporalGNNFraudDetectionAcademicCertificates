"""
data/temporal_dataset.py
------------------------
PyTorch Dataset that wraps the parquet files into a sequence of HeteroData
snapshots ready for model consumption.
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from graph.heterogeneous_graph import build_hetero_graph
from graph.snapshots import iter_snapshots

log = logging.getLogger(__name__)


class TemporalCredentialDataset(Dataset):
    """
    Loads the synthetic (or normalised real) parquet files and returns
    one HeteroData per temporal snapshot.

    Parameters
    ----------
    data_dir : str
        Directory containing students/institutions/verifiers/
        credentials/events parquet files.
    n_snapshots : int
        Total number of temporal snapshots (default 60).
    """

    def __init__(self, data_dir: str, n_snapshots: int = 60) -> None:
        super().__init__()
        p = Path(data_dir)
        log.info("Loading parquet files from %s …", data_dir)
        self.students     = pd.read_parquet(p / "students.parquet")
        self.institutions = pd.read_parquet(p / "institutions.parquet")
        self.verifiers    = pd.read_parquet(p / "verifiers.parquet")
        self.credentials  = pd.read_parquet(p / "credentials.parquet")
        self.events       = pd.read_parquet(p / "events.parquet")
        self.n_snapshots  = n_snapshots

        log.info(
            "Dataset: %d students | %d institutions | %d verifiers | "
            "%d credentials | %d events",
            len(self.students), len(self.institutions), len(self.verifiers),
            len(self.credentials), len(self.events),
        )

        # Pre-build all snapshots
        log.info("Building %d HeteroData snapshots …", n_snapshots)
        self._snapshots: list = []
        for snap_id, snap_evts in iter_snapshots(self.events, n_snapshots):
            hetero = build_hetero_graph(
                snapshot_id=snap_id,
                all_events=self.events,
                students=self.students,
                institutions=self.institutions,
                verifiers=self.verifiers,
                credentials=self.credentials,
                snap_events=snap_evts,
            )
            self._snapshots.append(hetero)
        log.info("Snapshots built.")

    def __len__(self) -> int:
        return self.n_snapshots

    def __getitem__(self, idx: int):
        return self._snapshots[idx]

    def get_input_dims(self) -> dict[str, int]:
        """Return feature dimension per node type."""
        snap = self._snapshots[0]
        return {nt: snap[nt].x.shape[1] for nt in
                ["student", "institution", "verifier", "credential"]}
