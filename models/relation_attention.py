"""
models/relation_attention.py
-----------------------------
Relation-aware heterogeneous graph attention convolution.

For each edge type (src_type, rel, tgt_type), we maintain:
  - W_r  : relation-specific linear transformation
  - a_r  : relation-specific attention vector

Update rule for node v of type t:
    h_v^(l+1) = activation(
        W_self * h_v^l
        + sum_r sum_{u in N_r(v)} alpha_{uv}^r * W_r * h_u^l
    )

Attention:
    e_{uv}^r = LeakyReLU(a_r^T [W_r h_u || W_r h_v])
    alpha_{uv}^r = softmax over N_r(v) of e_{uv}^r
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from graph.schema import EDGE_TYPES


class RelationAwareConv(nn.Module):
    """
    One layer of relation-aware heterogeneous graph attention convolution.

    Parameters
    ----------
    in_dims   : dict  node_type -> input feature dim
    out_dim   : int   output embedding dim (same for all node types)
    edge_types: list of (src_type, rel, tgt_type)
    dropout   : float attention dropout
    slope     : float LeakyReLU negative slope
    """

    def __init__(
        self,
        in_dims: dict[str, int],
        out_dim: int,
        edge_types: list[tuple[str, str, str]] = EDGE_TYPES,
        dropout: float = 0.3,
        slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.out_dim    = out_dim
        self.edge_types = edge_types
        self.dropout    = nn.Dropout(dropout)
        self.slope      = slope

        # Self-transformation per node type
        all_types = set()
        for s, _, t in edge_types:
            all_types.add(s); all_types.add(t)
        self.W_self = nn.ModuleDict({
            nt: nn.Linear(in_dims[nt], out_dim, bias=False)
            for nt in all_types if nt in in_dims
        })

        # Relation-specific transformation + attention vector
        self.W_r = nn.ModuleDict({
            rel: nn.Linear(in_dims[src], out_dim, bias=False)
            for src, rel, _ in edge_types if src in in_dims
        })
        self.a_r = nn.ParameterDict({
            rel: nn.Parameter(torch.empty(2 * out_dim))
            for _, rel, _ in edge_types
        })
        for p in self.a_r.values():
            nn.init.xavier_uniform_(p.unsqueeze(0))

        self.act = nn.ELU()

    def forward(self, x_dict: dict[str, torch.Tensor],
                data: HeteroData) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x_dict : node_type -> (N_type, in_dim) tensor
        data   : HeteroData with edge_index per edge type

        Returns
        -------
        dict   : node_type -> (N_type, out_dim) tensor
        """
        # Self-aggregations initialised
        out: dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.W_self:
                out[nt] = self.W_self[nt](x)

        # Relation-wise message passing
        for (src_type, rel, tgt_type) in self.edge_types:
            if src_type not in x_dict or tgt_type not in out:
                continue
            ei = data[src_type, rel, tgt_type].edge_index
            if ei is None or ei.shape[1] == 0:
                continue

            src_idx, tgt_idx = ei[0], ei[1]
            h_src = self.W_r[rel](x_dict[src_type][src_idx])   # (E, out_dim)
            h_tgt = out[tgt_type][tgt_idx]                      # (E, out_dim)

            # Attention
            cat = torch.cat([h_src, h_tgt], dim=-1)             # (E, 2*out_dim)
            e   = F.leaky_relu((cat * self.a_r[rel]).sum(-1),
                               negative_slope=self.slope)        # (E,)

            # Scatter softmax
            N_tgt = out[tgt_type].shape[0]
            alpha = torch.zeros(ei.shape[1], device=e.device)
            for i in range(N_tgt):
                mask = tgt_idx == i
                if mask.any():
                    alpha[mask] = torch.softmax(e[mask], dim=0)
            alpha = self.dropout(alpha)

            # Aggregate
            msg = alpha.unsqueeze(-1) * h_src                   # (E, out_dim)
            out[tgt_type] = out[tgt_type].index_add(0, tgt_idx, msg)

        return {nt: self.act(h) for nt, h in out.items()}
