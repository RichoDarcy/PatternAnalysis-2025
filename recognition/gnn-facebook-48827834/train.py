import torch
import torch.nn.functional as F
from dataset import build_data, split
from modules import GNN

def train(seed=1, hidden=128, dropout=0.5, lr=1e-2, weight_decay=5e-4, epochs=50):
    g = split(build_data(), seed=seed)
    num_classes = int(g.y.max().item()) + 1
    model = GNN(g.num_node_features, hidden, num_classes, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(g.x, g.edge_index, use_dropout=True)
        loss = F.cross_entropy(logits[g.train_mask], g.y[g.train_mask])
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"[v1] epoch={epoch:03d} | loss={loss.item():.5f}")

if __name__ == "__main__":
    train()