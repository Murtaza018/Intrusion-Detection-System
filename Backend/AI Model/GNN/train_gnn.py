import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pickle
import os
import numpy as np
from tqdm import tqdm

# --- PRODUCTION CONFIG ---
EMBEDDING_DIM = 16
NUM_FEATURES = 36 
EPOCHS = 10
LR = 0.001
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

def train_production():
    files = [f for f in os.listdir(SNAPSHOT_PATH) if f.endswith('.pkl')]
    model = ContextSAGE(in_channels=NUM_FEATURES, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Scheduler: Reduces LR by half every 3 epochs to refine the "Context"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    print(f"[*] Starting Production Training (10 Epochs) on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        np.random.shuffle(files) # Shuffle to prevent order-bias
        
        pbar = tqdm(files, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for filename in pbar:
            try:
                with open(os.path.join(SNAPSHOT_PATH, filename), 'rb') as f:
                    data_dict = pickle.load(f)
                
                edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long).to(DEVICE)
                edge_attr = torch.tensor(data_dict['edge_attr'], dtype=torch.float).to(DEVICE)
                
                # Align Node Features
                x = torch.zeros((data_dict['node_count'], NUM_FEATURES)).to(DEVICE)
                x.index_add_(0, edge_index[0], edge_attr[:, :NUM_FEATURES])
                
                optimizer.zero_grad()
                z = model(x, edge_index)
                
                # Unsupervised Link Prediction
                pos_scores = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
                loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            except:
                continue
        
        scheduler.step()
        avg_loss = total_loss / len(files)
        print(f"[>] Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        
        # Save a backup after every epoch
        torch.save(model.state_dict(), f"gnn_context_engine_epoch_{epoch+1}.pth")

    # Final Save
    torch.save(model.state_dict(), "gnn_context_engine_final.pth")
    print("\n[***] PRODUCTION TRAINING COMPLETE! Final model: gnn_context_engine_final.pth")

if __name__ == "__main__":
    train_production()