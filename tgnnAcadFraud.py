# ==========================================
# Temporal GNN Fraud Detection for blockchain based acedemic credential
# Real Corilla + Synthetic Augmentation
# Colab-Optimized
# ==========================================

#!pip install torch-geometric -q

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 1. LOAD REAL CORILLA DATA
# ==========================================================

students_real = pd.read_csv("students.csv")        # student_id, registration_date
institutions_real = pd.read_csv("institutions.csv")# institution_id, name, accreditation_status
certificates_real = pd.read_csv("certificates.csv")# certificate_id, student_id, institution_id, issue_date, revocation_date

students_real = students_real.rename(columns={"student_id": "stud_id"})
institutions_real = institutions_real.rename(
    columns={"institution_id": "inst_id", "accreditation_status": "accredited"}
)
institutions_real["accredited"] = institutions_real["accredited"].astype(int)

certificates_real = certificates_real.rename(
    columns={"certificate_id": "cert_id", "student_id": "stud_id", "institution_id": "inst_id"}
)

certificates_real["issue_date"] = pd.to_datetime(certificates_real["issue_date"])
certificates_real["revoked"] = certificates_real["revocation_date"].notna().astype(int)

certificates_real["time"] = certificates_real["issue_date"].rank(method="dense").astype(int)

print("✔ Real Corilla data loaded")

# ==========================================================
# 2. SYNTHETIC DATA AUGMENTATION
# ==========================================================

SYN_RATIO = 0.4
num_syn = int(len(certificates_real) * SYN_RATIO)

students_syn = pd.DataFrame({
    "stud_id": range(
        students_real["stud_id"].max()+1,
        students_real["stud_id"].max()+1 + 50
    )
})

institutions_syn = pd.DataFrame({
    "inst_id": range(
        institutions_real["inst_id"].max()+1,
        institutions_real["inst_id"].max()+1 + 5
    ),
    "accredited": np.random.choice([0,1], 5, p=[0.3,0.7])
})

certificates_syn = pd.DataFrame({
    "cert_id": range(
        certificates_real["cert_id"].max()+1,
        certificates_real["cert_id"].max()+1 + num_syn
    ),
    "stud_id": np.random.choice(students_syn["stud_id"], num_syn),
    "inst_id": np.random.choice(institutions_syn["inst_id"], num_syn),
    "revoked": np.random.choice([0,1], num_syn, p=[0.85,0.15]),
    "time": np.random.randint(
        certificates_real["time"].max()-5,
        certificates_real["time"].max()+5,
        num_syn
    )
})

print("✔ Synthetic augmentation generated")

# ==========================================================
# 3. MERGE HYBRID DATA
# ==========================================================

students = pd.concat([students_real, students_syn], ignore_index=True)
institutions = pd.concat([institutions_real, institutions_syn], ignore_index=True)
certificates = pd.concat([certificates_real, certificates_syn], ignore_index=True)

TIMESTEPS = certificates["time"].max()

print(f"✔ Hybrid dataset ready | Total certificates: {len(certificates)}")

# ==========================================================
# 4. BUILD TEMPORAL GRAPHS
# ==========================================================

graphs = {}

for t in range(1, TIMESTEPS+1):
    subset = certificates[
        (certificates["time"] <= t) &
        (certificates["time"] > t-5)
    ]

    G = nx.Graph()

    for _, row in subset.iterrows():
        inst = f"I{row.inst_id}"
        stud = f"S{row.stud_id}"
        cert = f"C{row.cert_id}"

        acc = institutions[institutions.inst_id == row.inst_id]["accredited"].values[0]

        G.add_node(inst, type_id=0, accredited=acc)
        G.add_node(stud, type_id=1, accredited=0)
        G.add_node(cert, type_id=2, accredited=0, revoked=int(row.revoked))

        G.add_edge(inst, cert)
        G.add_edge(cert, stud)

    graphs[t] = G

print("✔ Temporal graphs constructed")

# ==========================================================
# 5. CONVERT TO PyTorch Geometric
# ==========================================================

pyg_graphs = []

for G in graphs.values():
    data = from_networkx(G)

    X, Y = [], []

    for n, d in G.nodes(data=True):
        deg = G.degree[n]
        X.append([d["type_id"], d["accredited"], deg])
        Y.append(d.get("revoked", 0))

    data.x = torch.tensor(X, dtype=torch.float)
    data.y = torch.tensor(Y, dtype=torch.long)
    pyg_graphs.append(data)

print("✔ PyG graphs ready")

# ==========================================================
# 6. TRAIN / TEST SPLIT
# ==========================================================

split = int(0.8 * len(pyg_graphs))
train_graphs = pyg_graphs[:split]
test_graphs = pyg_graphs[split:]

# ==========================================================
# 7. LIGHTWEIGHT GNN MODEL
# ==========================================================

class LightGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 8)
        self.lin = torch.nn.Linear(8, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

model = LightGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_fn = torch.nn.CrossEntropyLoss()

# ==========================================================
# 8. TRAINING
# ==========================================================

for epoch in range(15):
    model.train()
    total_loss = 0

    for data in train_graphs:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_graphs):.4f}")

# ==========================================================
# 9. EVALUATION
# ==========================================================

model.eval()
y_true, y_pred, y_score = [], [], []

for data in test_graphs:
    data = data.to(device)
    out = model(data.x, data.edge_index)

    probs = F.softmax(out, dim=1)[:,1].cpu().numpy()
    preds = out.argmax(dim=1).cpu().numpy()

    y_true.extend(data.y.cpu().numpy())
    y_pred.extend(preds)
    y_score.extend(probs)

print("\n--- FINAL RESULTS ---")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1-score :", f1_score(y_true, y_pred))
print("ROC-AUC  :", roc_auc_score(y_true, y_score))

# ==========================================================
# 10. SAMPLE FRAUD ALERTS
# ==========================================================

alerts = [f"Node {i} flagged as fraudulent" for i,p in enumerate(y_pred) if p == 1]
print("\nSample Alerts:", alerts[:10])
