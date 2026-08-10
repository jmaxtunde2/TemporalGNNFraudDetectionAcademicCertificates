"""
graph/schema.py
---------------
Node-type and relation-type definitions for the heterogeneous credential graph.

Node types:
    student      – a person enrolled in an educational programme
    institution  – a credential-issuing body
    credential   – an academic certificate / qualification record
    verifier     – an authorised agent who can validate credentials

NOTE: Blockchain transactions are NOT represented as graph nodes in this
implementation. See IMPLEMENTATION_NOTES.md §IN-01.
"""

from enum import IntEnum


class NodeType(IntEnum):
    STUDENT = 0
    INSTITUTION = 1
    CREDENTIAL = 2
    VERIFIER = 3


class RelationType(IntEnum):
    ISSUES = 0       # institution  → credential
    OWNS = 1         # student      → credential
    VERIFIES = 2     # verifier     → credential
    CO_ISSUES = 3    # institution  → institution  (joint / cross issuance)
    REVOKES = 4      # institution  → credential
    MODIFIES = 5     # institution  → credential


# Human-readable names
NODE_TYPE_NAMES: dict[int, str] = {
    NodeType.STUDENT: "student",
    NodeType.INSTITUTION: "institution",
    NodeType.CREDENTIAL: "credential",
    NodeType.VERIFIER: "verifier",
}

RELATION_NAMES: dict[int, str] = {
    RelationType.ISSUES: "issues",
    RelationType.OWNS: "owns",
    RelationType.VERIFIES: "verifies",
    RelationType.CO_ISSUES: "co_issues",
    RelationType.REVOKES: "revokes",
    RelationType.MODIFIES: "modifies",
}

# PyG edge-type tuples  (src_node_type, relation_name, dst_node_type)
EDGE_TYPES: list[tuple[str, str, str]] = [
    ("institution", "issues",    "credential"),
    ("student",     "owns",      "credential"),
    ("verifier",    "verifies",  "credential"),
    ("institution", "co_issues", "institution"),
    ("institution", "revokes",   "credential"),
    ("institution", "modifies",  "credential"),
]

# Valid fraud-type labels (used in validation)
FRAUD_TYPES = {
    "none",
    "fake_institution",
    "insider_corruption",
    "identity_substitution",
    "collusive_fraud",
    "retroactive_manipulation",
}
