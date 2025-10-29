
# train.py
import torch
import torch.nn.functional as F
from dataset import build_data, split
from modules import GNN

def indecis(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).flatten()




def num_classes(labels: torch.Tensor, unlabeled: int = -1) -> int:
    keep = labels[labels != unlabeled]
    return int(keep.max().item()) + 1 if keep.numel() else 0



def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()



def infer(model: GNN, graph, use_dropout: bool, n_classes: int) -> torch.Tensor:
    acts = model(graph.x, graph.edge_index, use_dropout=use_dropout)
    return acts if acts.shape[-1] == n_classes else model.out_linear(acts)



def train(seed=1, hidden=128, dropout=0.5, lr=1e-2, weight_decay=5e-4, epochs=200):
    g = split(build_data(), seed=seed)
    num_classes = num_classes(g.y, unlabeled=-1)
    model = GNN(g.num_node_features, hidden, num_classes, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_idx = indecis(g.train_mask)
    va_idx = indecis(g.val_mask)
    te_idx = indecis(g.test_mask)


    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits_tr = infer(model, g, use_dropout=True, n_classes=num_classes)
        loss = F.cross_entropy(logits_tr.index_select(0, tr_idx), g.y.index_select(0, tr_idx))
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_ev = infer(model, g, use_dropout=False, n_classes=num_classes)
            val_acc = accuracy(logits_ev.index_select(0, va_idx), g.y.index_select(0, va_idx))
            test_acc = accuracy(logits_ev.index_select(0, te_idx), g.y.index_select(0, te_idx))



        if epoch % 10 == 0:
            print(f"[v3] epoch={epoch:03d} | loss={loss:.5f} | val={val_acc:.4f} | test={test_acc:.4f}")



    torch.save(model.state_dict(), "gnn_facebook.pt")
    print("checkpoint saved: gnn_facebook.pt")

if __name__ == "__main__":
    train()
