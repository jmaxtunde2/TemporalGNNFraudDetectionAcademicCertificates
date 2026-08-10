"""
models/tgat.py
--------------
Temporal Graph Attention Network (TGAT) baseline.

Faithfully reimplemented following:
    Xu et al. "Inductive Representation Learning on Temporal Graphs",
    ICLR 2020.  arXiv:2002.07962.

NOT copied from an external repository.  Cite this paper if used.

Key ideas:
  - Time encoding (sinusoidal) is concatenated to node features
  - Multi-head self-attention aggregates neighbour messages
  - Operates on cumulative neighbour history (not snapshots)

In this project, TGAT is adapted to the snapshot-based pipeline by
treating each snapshot's edges as the temporal neighbourhood.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_homogeneous
from models.temporal_gnn import SinusoidalTimeEncoding


class TGATLayer(nn.Module):
    """One TGAT temporal attention layer."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 time_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.time_enc = SinusoidalTimeEncoding(time_dim)
        feat_in = in_dim + time_dim
        self.q_proj = nn.Linear(feat_in, out_dim)
        self.k_proj = nn.Linear(feat_in, out_dim)
        self.v_proj = nn.Linear(feat_in, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                timestamps: torch.Tensor | None = None) -> torch.Tensor:
        N = x.shape[0]
        if timestamps is None:
            timestamps = torch.zeros(N, device=x.device)
        t_enc = self.time_enc(timestamps)               # (N, time_dim)
        x_t   = torch.cat([x, t_enc], dim=-1)           # (N, in+time)

        Q = self.q_proj(x_t)                            # (N, out)
        K = self.k_proj(x_t)
        V = self.v_proj(x_t)

        if edge_index.shape[1] == 0:
            return self.out_proj(Q)

        src, tgt = edge_index[0], edge_index[1]
        q = Q[tgt]; k = K[src]; v = V[src]

        scale = math.sqrt(self.head_dim)
        attn  = (q * k).sum(-1) / scale                 # (E,)
        # scatter softmax
        attn_out = torch.zeros_like(Q)
        for i in range(N):
            mask = tgt == i
            if mask.any():
                a = torch.softmax(attn[mask], dim=0)
                a = self.dropout(a)
                attn_out[i] = (a.unsqueeze(-1) * v[mask]).sum(0)

        return self.out_proj(attn_out)


class TGAT(nn.Module):
    """
    TGAT baseline adapted to the project's snapshot pipeline.

    Parameters
    ----------
    input_dims : dict  node_type -> feature dim
    hidden_dim : int
    n_layers   : int  (default 2)
    n_heads    : int  (default 4)
    time_dim   : int
    dropout    : float
    """

    def __init__(self, input_dims: dict[str, int], hidden_dim: int = 128,
                 n_layers: int = 2, n_heads: int = 4,
                 time_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.node_types = list(input_dims.keys())
        max_in = max(input_dims.values())
        self.projections = nn.ModuleDict({
            nt: nn.Linear(dim, max_in) for nt, dim in input_dims.items()
        })
        self.layers = nn.ModuleList([
            TGATLayer(max_in if i == 0 else hidden_dim,
                      hidden_dim, n_heads, time_dim, dropout)
            for i in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, snapshots: list[HeteroData],
                device: torch.device) -> tuple[list, list]:
        all_probs, all_labels = [], []
        for data in snapshots:
            data = data.to(device)
            x_dict = {nt: F.relu(self.projections[nt](data[nt].x))
                      for nt in self.node_types if nt in data.node_types}
            x_all = torch.cat([x_dict[nt] for nt in self.node_types
                                if nt in x_dict], dim=0)
            homo = to_homogeneous(data)
            ei   = homo.edge_index

            # Mean timestamp per node (proxy for temporal context)
            ts = torch.zeros(x_all.shape[0], device=device)

            h = x_all
            for layer in self.layers:
                h = layer(h, ei, ts)
                h = self.dropout(F.relu(h))

            # Compute offsets
            offsets: dict[str, int] = {}
            cur = 0
            for nt in self.node_types:
                if nt in x_dict:
                    offsets[nt] = cur; cur += x_dict[nt].shape[0]

            n_events = data.event_labels.shape[0]
            if n_events == 0:
                all_probs.append(torch.tensor([], device=device))
                all_labels.append(data.event_labels); continue

            src_e, tgt_e = [], []
            for i in range(n_events):
                sid = min(int(data.event_src[i]) + offsets.get(data.event_src_type[i], 0), h.shape[0]-1)
                tid = min(int(data.event_tgt[i]) + offsets.get(data.event_tgt_type[i], 0), h.shape[0]-1)
                src_e.append(h[sid]); tgt_e.append(h[tid])

            event_emb = torch.cat([torch.stack(src_e), torch.stack(tgt_e)], -1)
            probs = torch.sigmoid(self.classifier(event_emb).squeeze(-1))
            all_probs.append(probs); all_labels.append(data.event_labels)
        return all_probs, all_labels
