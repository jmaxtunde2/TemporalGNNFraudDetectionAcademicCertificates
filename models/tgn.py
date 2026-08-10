"""
models/tgn.py
--------------
Temporal Graph Network (TGN) baseline.

Faithfully reimplemented following:
    Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic Graphs",
    arXiv:2006.10637, 2020.

NOT copied from an external repository.

Key components implemented:
  - Memory module: stores a per-node memory vector updated by a GRU
  - Message function: raw message = concat(src_mem, tgt_mem, time_enc, edge_feat)
  - Memory updater: GRU processes messages to update memory
  - Embedding: graph attention over temporal neighbours using memory

In the snapshot pipeline, memory is updated once per snapshot.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_homogeneous
from models.temporal_gnn import SinusoidalTimeEncoding


class TGNMemory(nn.Module):
    """Per-node memory updated by a GRU at each snapshot."""

    def __init__(self, n_nodes: int, mem_dim: int) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.register_buffer("memory", torch.zeros(n_nodes, mem_dim))
        self.updater = nn.GRUCell(mem_dim * 2, mem_dim)

    def get(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.memory[node_ids]

    def update(self, src: torch.Tensor, tgt: torch.Tensor,
               src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> None:
        msg = torch.cat([src, tgt], dim=-1)                # (E, 2*mem_dim)
        new_src = self.updater(msg, self.memory[src_ids])  # (E, mem_dim)
        self.memory[src_ids] = new_src.detach()


class TGN(nn.Module):
    """
    TGN baseline adapted to the snapshot pipeline.

    Memory is maintained across snapshots (temporal state).

    Parameters
    ----------
    input_dims : dict  node_type -> feature dim
    hidden_dim : int
    mem_dim    : int   memory dimension
    time_dim   : int
    dropout    : float
    """

    def __init__(self, input_dims: dict[str, int], hidden_dim: int = 128,
                 mem_dim: int = 128, time_dim: int = 64,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.node_types = list(input_dims.keys())
        max_in = max(input_dims.values())

        self.projections = nn.ModuleDict({
            nt: nn.Linear(dim, max_in) for nt, dim in input_dims.items()
        })
        self.time_enc  = SinusoidalTimeEncoding(time_dim)
        self.dropout   = nn.Dropout(dropout)

        # Embedding layer: combine raw features + memory
        self.embed = nn.Linear(max_in + mem_dim, hidden_dim)

        # Graph attention for neighbour aggregation
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)

        # Event classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))

        # Memory (size fixed to a large pool; indexed by global node id)
        self._mem_dim = mem_dim
        self._memory: dict[str, torch.Tensor] = {}

    def _get_memory(self, node_type: str, n_nodes: int,
                    device: torch.device) -> torch.Tensor:
        if node_type not in self._memory or \
           self._memory[node_type].shape[0] != n_nodes:
            self._memory[node_type] = torch.zeros(n_nodes, self._mem_dim, device=device)
        return self._memory[node_type].to(device)

    def _update_memory(self, node_type: str, ids: torch.Tensor,
                       new_state: torch.Tensor) -> None:
        self._memory[node_type][ids] = new_state.detach()

    def forward(self, snapshots: list[HeteroData],
                device: torch.device) -> tuple[list, list]:
        all_probs, all_labels = [], []

        for data in snapshots:
            data = data.to(device)

            # Project raw features
            x_dict = {nt: F.relu(self.projections[nt](data[nt].x))
                      for nt in self.node_types if nt in data.node_types}

            # Combine with memory
            h_dict: dict[str, torch.Tensor] = {}
            for nt, x in x_dict.items():
                mem = self._get_memory(nt, x.shape[0], device)
                h   = torch.cat([x, mem], dim=-1)
                h_dict[nt] = F.relu(self.embed(h))

            x_all  = torch.cat([h_dict[nt] for nt in self.node_types
                                 if nt in h_dict], dim=0)
            homo   = to_homogeneous(data)
            ei     = homo.edge_index

            # Self-attention aggregation
            if ei.shape[1] > 0:
                Q = self.attn_q(x_all)
                K = self.attn_k(x_all)
                V = self.attn_v(x_all)
                src, tgt = ei[0], ei[1]
                attn = (Q[tgt] * K[src]).sum(-1) / (Q.shape[-1] ** 0.5)
                N = x_all.shape[0]
                out = torch.zeros_like(x_all)
                for i in range(N):
                    mask = tgt == i
                    if mask.any():
                        a = torch.softmax(attn[mask], dim=0)
                        out[i] = (a.unsqueeze(-1) * V[src[mask]]).sum(0)
                x_all = x_all + self.dropout(out)

            # Update memory with mean representation
            offsets: dict[str, int] = {}
            cur = 0
            for nt in self.node_types:
                if nt in h_dict:
                    n = h_dict[nt].shape[0]
                    new_state = x_all[cur:cur + n]
                    self._update_memory(nt, torch.arange(n, device=device), new_state)
                    offsets[nt] = cur; cur += n

            # Event classifier
            n_events = data.event_labels.shape[0]
            if n_events == 0:
                all_probs.append(torch.tensor([], device=device))
                all_labels.append(data.event_labels); continue

            src_e, tgt_e = [], []
            for i in range(n_events):
                sid = min(int(data.event_src[i]) + offsets.get(data.event_src_type[i], 0), x_all.shape[0]-1)
                tid = min(int(data.event_tgt[i]) + offsets.get(data.event_tgt_type[i], 0), x_all.shape[0]-1)
                src_e.append(x_all[sid]); tgt_e.append(x_all[tid])

            event_emb = torch.cat([torch.stack(src_e), torch.stack(tgt_e)], -1)
            probs = torch.sigmoid(self.classifier(event_emb).squeeze(-1))
            all_probs.append(probs); all_labels.append(data.event_labels)

        return all_probs, all_labels
