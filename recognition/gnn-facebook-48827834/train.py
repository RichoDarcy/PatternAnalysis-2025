
import torch
import torch.nn.functional as F
from dataset import build_data, split
from modules import GNN

def indecis(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).flatten()




def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train(seed=1, hidden=128, dropout=0.5, lr=1e-2, weight_decay=5e-4, epochs=200):
    g = split(build_data(), seed=seed)
    num_classes = int(g.y.max().item()) + 1
    model = GNN(g.num_node_features, hidden, num_classes, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_idx = indecis(g.train_mask)
    va_idx = indecis(g.val_mask)
    te_idx = indecis(g.test_mask)


    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits_tr = model(g.x, g.edge_index, use_dropout=True)
        loss = F.cross_entropy(logits_tr.index_select(0, tr_idx), g.y.index_select(0, tr_idx))
        loss.backward()
        opt.step()


        model.eval()
        with torch.no_grad():
            logits_ev = model(g.x, g.edge_index, use_dropout=False)
            val_acc = accuracy(logits_ev.index_select(0, va_idx), g.y.index_select(0, va_idx))
            test_acc = accuracy(logits_ev.index_select(0, te_idx), g.y.index_select(0, te_idx))

        if epoch % 10 == 0:
            print(f"[v2] epoch={epoch:03d} | loss={loss:.5f} | val={val_acc:.4f} | test={test_acc:.4f}")

if __name__ == "__main__":
    train()
