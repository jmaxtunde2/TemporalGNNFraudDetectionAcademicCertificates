"""
graph/relations.py
------------------
Relation-ID constants and the learnable relation-embedding module.
"""

import torch
import torch.nn as nn
from graph.schema import RelationType, RELATION_NAMES

# Dict mapping name → int id  (for use in data generation)
RELATIONS: dict[str, int] = {name: int(rid) for rid, name in RELATION_NAMES.items()}
N_RELATIONS: int = len(RELATIONS)


class RelationEmbedding(nn.Module):
    """
    Learnable embedding table for relation types.

    Parameters
    ----------
    embedding_dim : int
        Size of each relation embedding vector.
    n_relations : int
        Number of distinct relation types (default: 6).
    """

    def __init__(self, embedding_dim: int, n_relations: int = N_RELATIONS) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_relations, embedding_dim)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        relation_ids : torch.Tensor  shape (E,)
            Integer relation-type ids for each event.

        Returns
        -------
        torch.Tensor  shape (E, embedding_dim)
        """
        return self.embedding(relation_ids)
