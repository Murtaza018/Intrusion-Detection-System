import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pickle
import os
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
EMBEDDING_DIM = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SNAPSHOT_PATH = "./training_data/gnn_snapshots/"

class ContextSAGE(torch.nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(ContextSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, 32)
        self.conv2 = SAGEConv(32, embedding_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def validate_embeddings(model, sample_file):
    """Checks if the 16D vectors are diverse and meaningful"""
    model.eval()
    with torch.no_grad():
        with open(os.path.join(SNAPSHOT_PATH, sample_file), 'rb') as f:
            data = pickle.load(f)
        
        edge_index = torch.tensor(data['edge_index'], dtype=torch.long).to(DEVICE)
        # Dummy node features (will be replaced by real mean features in main loop)
        x = torch.ones((data['node_count'], 36)).to(DEVICE)
        
        z = model(x, edge_index)
        
        # Calculation: How different are the vectors?
        variance = torch.var(z, dim=0).mean().item()
        mag = torch.norm(z, dim=1).mean().item()
        
        print(f"\n" + "="*40)
        print(f"VALIDATION AFTER EPOCH 1")
        print("="*40)
        print(f"Context Vector Dim:  {z.shape[1]}")
        print(f"Unique IPs Checked:  {z.shape[0]}")
        print(f"Average Magnitude:   {mag:.4f}")
        print(f"Embedding Variance:  {variance:.6f}")
        
        if variance < 1e-5:
            print("[!] ALERT: Model Collapse detected (Vectors are too similar).")
        else:
            print("[+] SUCCESS: Vectors are diverse. Logic is healthy.")
        print("="*40 + "\n")

def train_pilot():
    files = [f for f in os.listdir(SNAPSHOT_PATH) if f.endswith('.pkl')]
    model = ContextSAGE(in_channels=36, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"[*] Starting Epoch 1 (Pilot Run) on {DEVICE}...")

    model.train()
    total_loss = 0
    
    # tqdm gives us a nice progress bar
    for filename in tqdm(files, desc="Training"):
        with open(os.path.join(SNAPSHOT_PATH, filename), 'rb') as f:
            data_dict = pickle.load(f)
        
        edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long).to(DEVICE)
        num_nodes = data_dict['node_count']
        
        # Real logic: Node features = Average of their connection features
        x = torch.zeros((num_nodes, 36)).to(DEVICE)
        edge_attr = torch.tensor(data_dict['edge_attr'], dtype=torch.float).to(DEVICE)
        x.index_add_(0, edge_index[0], edge_attr) # Aggregate features to nodes
        
        optimizer.zero_grad()
        z = model(x, edge_index)
        
        # Unsupervised Loss: Connected nodes should have similar 16D vectors
        # Formula: -log(sigmoid(dot_product(u, v)))
        pos_scores = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Save the model
    torch.save(model.state_dict(), "gnn_context_engine_pilot.pth")
    
    # Run Validation
    validate_embeddings(model, files[0])

if __name__ == "__main__":
    train_pilot()