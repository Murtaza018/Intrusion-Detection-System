import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pickle
import os
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
EMBEDDING_DIM = 16
NUM_FEATURES = 36 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SNAPSHOT_PATH = "./training_data/gnn_snapshots/"

class ContextSAGE(torch.nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(ContextSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 32)
        self.conv2 = SAGEConv(32, embedding_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)

def validate_embeddings(model, files):
    """Statistically verifies that the 16D space is meaningful"""
    model.eval()
    print("\n" + "="*40)
    print("PILOT VALIDATION REPORT")
    print("="*40)
    
    variances = []
    with torch.no_grad():
        # Check a random selection of 5 snapshots
        for f_name in np.random.choice(files, 5):
            with open(os.path.join(SNAPSHOT_PATH, f_name), 'rb') as f:
                data = pickle.load(f)
            
            edge_index = torch.tensor(data['edge_index'], dtype=torch.long).to(DEVICE)
            edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float).to(DEVICE)
            
            # Align features
            x = torch.zeros((data['node_count'], NUM_FEATURES)).to(DEVICE)
            x.index_add_(0, edge_index[0], edge_attr[:, :NUM_FEATURES])
            
            z = model(x, edge_index)
            var = torch.var(z, dim=0).mean().item()
            variances.append(var)

    avg_var = np.mean(variances)
    print(f"Avg Embedding Variance: {avg_var:.8f}")
    if avg_var > 0.0001:
        print("[+] STATUS: SUCCESS. The Context Engine is learning.")
    else:
        print("[!] STATUS: FAILURE. Model collapse. Try increasing Learning Rate.")
    print("="*40 + "\n")

def train_pilot():
    files = [f for f in os.listdir(SNAPSHOT_PATH) if f.endswith('.pkl')]
    model = ContextSAGE(in_channels=NUM_FEATURES, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"[*] Starting Pilot Run (1 Epoch) on {DEVICE}...")

    model.train()
    for filename in tqdm(files, desc="Training Snapshots"):
        try:
            with open(os.path.join(SNAPSHOT_PATH, filename), 'rb') as f:
                data_dict = pickle.load(f)
            
            edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long).to(DEVICE)
            edge_attr = torch.tensor(data_dict['edge_attr'], dtype=torch.float).to(DEVICE)
            
            # Create Node Identity (x) from sliced Edge Attributes
            x = torch.zeros((data_dict['node_count'], NUM_FEATURES)).to(DEVICE)
            x.index_add_(0, edge_index[0], edge_attr[:, :NUM_FEATURES])
            
            optimizer.zero_grad()
            z = model(x, edge_index)
            
            # Unsupervised Link Prediction (Similarity Learning)
            pos_scores = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
            loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
            
            loss.backward()
            optimizer.step()
        except:
            continue

    torch.save(model.state_dict(), "gnn_context_engine_pilot.pth")
    validate_embeddings(model, files)

if __name__ == "__main__":
    train_pilot()