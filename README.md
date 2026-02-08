📜 Temporal Graph Neural Network for Academic Credential Fraud Detection

This repository implements a Hybrid Temporal Graph Neural Network (T-GNN) framework for detecting fraud in blockchain-based academic credential ecosystems.
The system combines real-world credential data (e.g., Corilla exports) with synthetic augmentation to model evolving fraud patterns involving students, institutions, certificates, and verifiers.

🔍 Motivation

Blockchain ensures immutability, but does not guaranty that data stored is genuine.

Fraudulent academic credentials may still be issued, verified, or stored immutably. These fraud patterns often:

Emerge gradually over time

Involve collusive behavior across multiple entities

Cannot be detected by document-level AI alone

This project introduces a Temporal Graph Intelligence layer on top of blockchain credentials to detect:

Diploma mills

Insider corruption

Identity substitution

Coordinated credential fraud

🧠 Key Contributions

Temporal Graph Modeling of academic ecosystems

Hybrid data pipeline (real Corilla + synthetic augmentation)

Node-level fraud detection with time-aware reasoning

Research-ready experimental design aligned with journal standards

🏗️ System Overview

The system models academic credentials as a temporal heterogeneous graph:

Nodes

Students

Institutions

Certificates

Edges

Institution → Certificate (issuance)

Certificate → Student (ownership)

Time

Graph snapshots constructed over sliding windows

A Graph Neural Network (GNN) learns structural patterns at each time step, while temporal aggregation captures evolving fraud behavior.

🧪 Supported Fraud Detection Tasks

Node-level classification

Fraudulent certificates

Malicious institutions

Temporal anomaly detection

Suspicious issuance bursts

Retroactive revocations

Predictive detection

Early warning of future fraud

🔬 Research Context

This repository accompanies research on:

Temporal Graph Neural Networks for Blockchain-Based Academic Credential Fraud Detection

It is designed to be:

Reproducible

Extensible

Suitable for academic benchmarking

🚀 Future Work

Planned extensions include:

Explainable AI (XAI) for graph decisions

Multimodal document + graph fusion

Federated learning across institutions

Real-time blockchain smart contract triggers

Cross-border credential fraud detection

⚠️ Limitations

Partial reliance on synthetic data

No document-content analysis (yet)

Requires sufficient historical data for optimal performance

🤝 Contributing

Contributions are welcome!

Open issues for bugs or ideas

Submit pull requests for improvements

Extend the framework to new domains
