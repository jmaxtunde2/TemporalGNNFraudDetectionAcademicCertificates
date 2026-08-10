"""
models/temporal_gnn.py
-----------------------
Full T-GNN model:

  snapshot_t  →  node-type projections  →  2× RelationAwareConv
              →  GRU(H_t, hidden_{t-1})  →  Z_t
  event (u,r,v,t) → concat(Z_t[u], Z_t[v], rel_emb[r], time_enc(t))
                  → MLP → sigmoid → fraud_probability

Output: binary fraud probability ∈ (0,1)  per event.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from models.relation_attention import RelationAwareConv
from graph.relations import RelationEmbedding, N_RELATIONS
from graph.schema import EDGE_TYPES


# ── Sinusoidal time encoding (TGAT-style) ─────────────────────────────────────
class SinusoidalTimeEncoding(nn.Module):
    """
    φ(t)[2i]   = sin(t / 10000^(2i/d))
    φ(t)[2i+1] = cos(t / 10000^(2i/d))
    Ref: Xu et al. 2020 (TGAT), ICLR.
    """
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float) *
            (-math.log(10000.0) / d)
        )
        self.register_buffer("div", div)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (E,) float seconds  →  (E, d)"""
        t = t.unsqueeze(-1)                                # (E,1)
        enc = torch.zeros(t.shape[0], self.d, device=t.device)
        enc[:, 0::2] = torch.sin(t * self.div)
        enc[:, 1::2] = torch.cos(t * self.div[:self.d // 2])
        return enc


class ScalarDeltaTimeEncoding(nn.Module):
    """Simple normalised time-delta scalar expanded to a vector via a linear."""
    def __init__(self, d: int, max_time: float = 600 * 86400) -> None:
        super().__init__()
        self.max_time = max_time
        self.linear   = nn.Linear(1, d)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_norm = (t / self.max_time).unsqueeze(-1)         # (E,1)
        return self.linear(t_norm)                          # (E,d)


# ── T-GNN ─────────────────────────────────────────────────────────────────────
class TemporalHeteroGNN(nn.Module):
    """
    Temporal Heterogeneous GNN for event-level fraud detection.

    Parameters
    ----------
    input_dims    : dict  node_type -> raw feature dim
    embedding_dim : int   shared node embedding dimension (default 128)
    gru_hidden    : int   GRU hidden state dimension (default 128)
    n_layers      : int   number of RelationAwareConv layers (default 2)
    dropout       : float dropout probability
    time_enc      : str   'sinusoidal' or 'scalar_delta'
    time_enc_dim  : int   time encoding output dimension
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        embedding_dim: int = 128,
        gru_hidden: int    = 128,
        n_layers: int      = 2,
        dropout: float     = 0.3,
        time_enc: str      = "sinusoidal",
        time_enc_dim: int  = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gru_hidden    = gru_hidden
        self.node_types    = list(input_dims.keys())
        self.dropout       = nn.Dropout(dropout)

        # ── Input projections per node type ──────────────
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(dim, embedding_dim)
            for nt, dim in input_dims.items()
        })

        # ── Relation-aware GNN layers ──────────────────
        dims_after_proj = {nt: embedding_dim for nt in input_dims}
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(
                RelationAwareConv(dims_after_proj, embedding_dim,
                                  dropout=dropout)
            )

        # ── GRU: processes the per-node GNN output over time ──
        # We pool all node embeddings into a global context vector
        self.gru = nn.GRUCell(embedding_dim, gru_hidden)

        # ── Relation embedding ─────────────────────────
        self.rel_emb = RelationEmbedding(embedding_dim)

        # ── Time encoding ─────────────────────────────
        if time_enc == "sinusoidal":
            self.time_enc = SinusoidalTimeEncoding(time_enc_dim)
        else:
            self.time_enc = ScalarDeltaTimeEncoding(time_enc_dim)
        self.time_enc_dim = time_enc_dim

        # ── Event-level classifier MLP ─────────────────
        # input: src_emb + tgt_emb + rel_emb + time_enc
        clf_in = gru_hidden + gru_hidden + embedding_dim + time_enc_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _project_nodes(self, data: HeteroData) -> dict[str, torch.Tensor]:
        return {
            nt: F.relu(self.input_proj[nt](data[nt].x))
            for nt in self.node_types if nt in data.node_types
        }

    def _apply_gnn(self, x_dict: dict[str, torch.Tensor],
                   data: HeteroData) -> dict[str, torch.Tensor]:
        for layer in self.gnn_layers:
            x_dict = layer(x_dict, data)
            x_dict = {nt: self.dropout(h) for nt, h in x_dict.items()}
        return x_dict

    def _pool_global(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean-pool all node embeddings into one global vector."""
        vecs = [h.mean(0) for h in x_dict.values() if h.shape[0] > 0]
        return torch.stack(vecs).mean(0)                    # (embedding_dim,)

    def _lookup_node_emb(self, node_embs: dict[str, torch.Tensor],
                         gru_out: torch.Tensor,
                         ids: torch.Tensor,
                         types: list[str],
                         device: torch.device) -> torch.Tensor:
        """
        For each event, look up the GRU-evolved embedding for the
        source or target node.  Falls back to zeros if node not found.
        """
        results = []
        for i, (nid, ntype) in enumerate(zip(ids.tolist(), types)):
            if ntype in node_embs and int(nid) < node_embs[ntype].shape[0]:
                results.append(node_embs[ntype][int(nid)])
            else:
                results.append(torch.zeros(self.gru_hidden, device=device))
        return torch.stack(results)                          # (E, gru_hidden)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        snapshots: list[HeteroData],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Process the full snapshot sequence through GNN + GRU.

        Returns
        -------
        all_probs  : list of (E_t,) fraud probability tensors per snapshot
        all_labels : list of (E_t,) ground-truth label tensors per snapshot
        """
        hidden: torch.Tensor | None = None
        # GRU hidden is a single (gru_hidden,) vector (no batch dim here)
        # We maintain per-node hidden states implicitly via the global pool trick.

        all_probs:  list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for data in snapshots:
            data = data.to(device)

            # 1. Project node features
            x_dict = self._project_nodes(data)

            # 2. Relation-aware GNN
            x_dict = self._apply_gnn(x_dict, data)

            # 3. GRU over global pooled representation
            global_vec = self._pool_global(x_dict)          # (emb_dim,)
            if hidden is None:
                hidden = torch.zeros(self.gru_hidden, device=device)
            hidden = self.gru(global_vec.unsqueeze(0),
                              hidden.unsqueeze(0)).squeeze(0)  # (gru_hidden,)

            # 4. Build per-node GRU-scaled embeddings
            # Scale each node embedding by the GRU hidden (broadcast)
            gru_embs: dict[str, torch.Tensor] = {}
            for nt, h in x_dict.items():
                # project node embeddings to gru_hidden via a simple linear
                gru_embs[nt] = h + hidden.unsqueeze(0).expand_as(
                    h[:, :self.gru_hidden])  # residual

            # 5. Event-level classifier
            n_events = data.event_labels.shape[0]
            if n_events == 0:
                all_probs.append(torch.tensor([], device=device))
                all_labels.append(torch.tensor([], device=device))
                continue

            src_emb = self._lookup_node_emb(
                gru_embs, hidden,
                data.event_src, data.event_src_type, device)  # (E, gru_hidden)
            tgt_emb = self._lookup_node_emb(
                gru_embs, hidden,
                data.event_tgt, data.event_tgt_type, device)  # (E, gru_hidden)

            rel_emb  = self.rel_emb(data.event_relation)       # (E, emb_dim)
            time_enc = self.time_enc(data.event_timestamp)      # (E, time_enc_dim)

            event_emb = torch.cat([src_emb, tgt_emb, rel_emb, time_enc], dim=-1)
            logits    = self.classifier(event_emb).squeeze(-1)  # (E,)
            probs     = torch.sigmoid(logits)

            all_probs.append(probs)
            all_labels.append(data.event_labels)

        return all_probs, all_labels
