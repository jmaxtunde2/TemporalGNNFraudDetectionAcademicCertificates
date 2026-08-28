"""
models/static_gcn.py
---------------------
Static homogeneous GCN baseline.

All node types are projected to a common feature space, then two
GCNConv layers are applied. No GRU, no temporal state.

This is the A1 ablation and the "Static GCN" baseline.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData


class StaticGCN(nn.Module):
    """
    Baseline: homogeneous 2-layer GCN with event-level binary classifier.

    Parameters
    ----------
    input_dims    : dict  node_type -> raw feature dim
    hidden_dim    : int
    dropout       : float
    """

    def __init__(self,
                 input_dims: dict[str, int],
                 hidden_dim: int = 128,
                 dropout: float  = 0.3) -> None:
        super().__init__()
        self.node_types = list(input_dims.keys())
        self.dropout    = nn.Dropout(dropout)

        # Project each node type to a common dim before GCN
        total_in = max(input_dims.values())
        self.projections = nn.ModuleDict({
            nt: nn.Linear(dim, total_in)
            for nt, dim in input_dims.items()
        })

        self.conv1 = GCNConv(total_in,   hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Classifier: src + tgt node embeddings → fraud prob
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _project(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {nt: F.relu(self.projections[nt](x))
                for nt, x in x_dict.items() if nt in self.projections}

    def forward(self, snapshots: list[HeteroData],
                device: torch.device) -> tuple[list, list]:
        all_probs, all_labels = [], []

        for data in snapshots:
            data = data.to(device)

            # Project to common space
            x_dict = {nt: F.relu(self.projections[nt](data[nt].x))
                      for nt in self.node_types if nt in data.node_types}

            # Convert hetero → homogeneous
            homo = data.to_homogeneous()
            x_all = torch.cat([x_dict[nt] for nt in self.node_types
                                if nt in x_dict], dim=0)

            if homo.edge_index.shape[1] == 0:
                all_probs.append(torch.tensor([], device=device))
                all_labels.append(data.event_labels)
                continue

            h = F.relu(self.conv1(x_all, homo.edge_index))
            h = self.dropout(h)
            h = F.relu(self.conv2(h, homo.edge_index))

            # Build offset map for node-type lookups
            offsets: dict[str, int] = {}
            cur = 0
            for nt in self.node_types:
                if nt in x_dict:
                    offsets[nt] = cur
                    cur += x_dict[nt].shape[0]

            n_events = data.event_labels.shape[0]
            if n_events == 0:
                all_probs.append(torch.tensor([], device=device))
                all_labels.append(data.event_labels)
                continue

            src_embs, tgt_embs = [], []
            for i in range(n_events):
                stype = data.event_src_type[i]
                ttype = data.event_tgt_type[i]
                sid = int(data.event_src[i]) + offsets.get(stype, 0)
                tid = int(data.event_tgt[i]) + offsets.get(ttype, 0)
                sid = min(sid, h.shape[0] - 1)
                tid = min(tid, h.shape[0] - 1)
                src_embs.append(h[sid])
                tgt_embs.append(h[tid])

            src_embs = torch.stack(src_embs)
            tgt_embs = torch.stack(tgt_embs)
            event_emb = torch.cat([src_embs, tgt_embs], dim=-1)
            probs = torch.sigmoid(self.classifier(event_emb).squeeze(-1))

            all_probs.append(probs)
            all_labels.append(data.event_labels)

        return all_probs, all_labels
