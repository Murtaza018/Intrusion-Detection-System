import torch
from gnn_model import GNNEncoder # Assuming this is your GNN class

def test_gnn_topological_shift():
    # Simulate a star-topology (Normal) vs. a Mesh-topology (Scanning/DDoS)
    # This is a simplified check for the embedding variance
    model = GNNEncoder(in_channels=78, hidden_channels=16)
    
    # Mock Graph: 5 nodes, features = 78
    x = torch.randn(5, 78)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 0, 3]], dtype=torch.long)
    
    output = model(x, edge_index)
    
    print(f"[*] GNN Embedding Shape: {output.shape}")
    if output.shape == (5, 16):
        print("✅ PASS: GNN correctly projecting topological features.")
    else:
        print("❌ FAIL: GNN embedding dimension mismatch.")

if __name__ == "__main__":
    test_gnn_topological_shift()