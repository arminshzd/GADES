import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mdtraj as md
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ========== PARAMETERS ==========
GRO_FILE = 'topology.gro'
XTC_FILE = 'trajectory.xtc'
PSI_FILE = 'psi.npy'   # shape: (n_frames, n_eigen)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
N_EPOCHS = 20
LEARNING_RATE = 1e-3

# ========== 1. Featurize trajectory ==========

def featurize_frame(traj, atom_indices=None):
    """
    Use pairwise distances as features.
    """
    if atom_indices is None:
        atom_indices = list(range(traj.n_atoms))
    
    pairs = [(i, j) for i in atom_indices for j in atom_indices if i < j]
    feat = md.compute_distances(traj, pairs)
    return feat

print("Loading trajectory...")
traj = md.load(XTC_FILE, top=GRO_FILE)
features = featurize_frame(traj)  # shape: (n_frames, n_features)

print("Loading targets...")
psi = np.load(PSI_FILE)  # shape: (n_frames, n_eigen)

# ========== 2. Dataset and DataLoader ==========

class TrajDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(features, psi, test_size=0.2, random_state=42)

train_ds = TrajDataset(X_train, y_train)
test_ds = TrajDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ========== 3. Define NN Model ==========

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = FeedForwardNN(input_dim=features.shape[1], output_dim=psi.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# ========== 4. Training Loop ==========

print("Training model...")
for epoch in range(N_EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in tqdm(train_loader):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch+1}] Train Loss: {running_loss / len(train_loader):.4f}")

# ========== 5. Evaluation ==========

model.eval()
with torch.no_grad():
    total_loss = 0.0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        total_loss += loss.item()
    print(f"Test MSE: {total_loss / len(test_loader):.4f}")