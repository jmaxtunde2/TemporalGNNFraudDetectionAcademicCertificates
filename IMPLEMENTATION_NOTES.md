# Implementation Notes

This file records every ambiguity found in the paper and the resolution
chosen during implementation. Per reviewer instructions, these are flagged
here rather than silently assumed.

---

## IN-01 — Blockchain Transactions as Graph Nodes

**Paper statement**: "The system records credential events on a blockchain
ledger; these transactions form part of the graph representation."

**Ambiguity**: It is unclear whether blockchain transaction records should
be fourth-class nodes in the heterogeneous graph, or merely metadata
associated with existing events.

**Resolution adopted**: Blockchain anchoring is represented as a boolean
feature column `on_chain` on each event row, **not** as a separate node
type. Reasons:
1. Adding a transaction node for every event would double the edge count
   without adding structural information not already in the event itself.
2. The paper's node-type table lists only `Student`, `Institution`,
   `Credential`, and `Verifier` as first-class nodes.
3. Fabricating a separate blockchain-node layer would constitute
   scientific embellishment.

If a future paper version explicitly requires blockchain nodes, add a
`BlockchainTx` node type to `graph/schema.py` and re-run the generator.

---

## IN-02 — Time Encoding Formula

**Paper statement**: "We encode temporal information using a time
encoding function φ(t)."

**Ambiguity**: The exact formula is not given (sinusoidal / linear /
learnable / time-delta scalar).

**Resolution adopted**: Sinusoidal encoding (following Xu et al. 2020,
TGAT) as the default, selectable via `configs/default.yaml`:

```
φ(t)[2i]   = sin(t / 10000^(2i/d))
φ(t)[2i+1] = cos(t / 10000^(2i/d))
```

where `d = model.time_encoding_dim` (default 64).

A simpler scalar `time_delta` mode is also available for ablation.

---

## IN-03 — Verifiers Absent from Real Corilla CSVs

**Paper statement**: Verifiers are listed as a core node type.

**Observation**: The existing `students.csv`, `institutions.csv`, and
`certificates.csv` contain no verifier table.

**Resolution adopted**:
- The synthetic generator always creates verifiers (5 per institution).
- `data/real_corilla_loader.py` emits a `UserWarning` when `verifiers.csv`
  is absent and synthesizes placeholder verifiers.
- The real-data path is clearly gated behind `--data-source real`.

---

## IN-04 — Five Fraud Scenarios vs. Five Output Classes

**Paper statement**: "We consider five fraud generation scenarios."

**Risk of misinterpretation**: A naive reader might implement a 5-class
softmax output.

**Resolution adopted**: The output is strictly **binary**
(`fraud_label ∈ {0, 1}`). The `fraud_type` column is metadata only and
is never used as a training target. This is enforced throughout the
codebase and validated by `data/validate_dataset.py`.

---

## IN-05 — Co-issuance Relation Semantics

**Paper statement**: Lists `co-issuance` as a relation type.

**Ambiguity**: Whether this is institution–institution, institution–
credential, or a hyperedge.

**Resolution adopted**: `co_issues` is modelled as an institution–
institution edge, created when two institutions jointly issue or
cross-validate the same credential. This is a natural interpretation
and avoids hyperedge complexity unsupported by PyG's `HeteroData`.

---

## IN-06 — TGN / TGAT Source Implementations

Both baselines are implemented from scratch following the original papers:

- **TGAT**: Xu et al., "Inductive Representation Learning on Temporal
  Graphs", ICLR 2020. arXiv:2002.07962.
- **TGN**: Rossi et al., "Temporal Graph Networks for Deep Learning on
  Dynamic Graphs", arXiv:2006.10637.

These are faithful lightweight reimplementations compatible with the
project's `HeteroData` pipeline. They are **not** copied from external
repositories. If PyG adds official TGN/TGAT support, prefer those.

---

## IN-07 — CNN Baseline

The paper lists CNN as a baseline. Since the input is a graph (not an
image), the CNN baseline operates on **feature matrices** extracted per
snapshot (flattened node-feature aggregations), treating the temporal
sequence of feature matrices as a 1-D convolution over the snapshot axis.
This is documented in `experiments/run_baselines.py`.

---

## IN-08 — Exact Event Count (200,000)

The generator distributes exactly 200,000 events as follows:

| Event type      | Count  |
|-----------------|--------|
| issuance        | 20,000 |
| ownership       | 20,000 |
| verification    | 120,000|
| revocation      | 10,000 |
| modification    | 10,000 |
| co_issuance     | 20,000 |
| **Total**       | **200,000** |

The exact split is configurable but the total is validated.
