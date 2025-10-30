import torch
import torch.nn.functional as F
from dataset import build_data, split
from modules import GNN

import os
import csv
from pathlib import Path
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_DIR = Path("runs"); RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = RUN_DIR / "gnn_facebook.pt"


def plotpath(dir_name: str = "runs") -> Path:
    p = Path(dir_name)
    p.mkdir(parents=True, exist_ok=True)
    return p

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
    g = split(build_data(), seed=seed).to(device)
    print("Running on:", (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"))
    n_classes = num_classes(g.y, unlabeled=-1)
    model = GNN(g.num_node_features, hidden, n_classes, dropout).to(device)


    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_idx = indecis(g.train_mask)
    va_idx = indecis(g.val_mask)
    te_idx = indecis(g.test_mask)
    run_dir = Path("runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    metrics_path.write_text("epoch,train_loss,val_acc,test_acc\n", encoding="utf-8")
    epoch_hist, loss_hist, val_hist, test_hist = [], [], [], []


    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits_tr = infer(model, g, use_dropout=True, n_classes=n_classes)
        loss = F.cross_entropy(logits_tr.index_select(0, tr_idx), g.y.index_select(0, tr_idx))
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_ev = infer(model, g, use_dropout=False, n_classes=n_classes)
            val_acc = accuracy(logits_ev.index_select(0, va_idx), g.y.index_select(0, va_idx))
            test_acc = accuracy(logits_ev.index_select(0, te_idx), g.y.index_select(0, te_idx))

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{loss.item():.6f},{val_acc:.6f},{test_acc:.6f}\n")
            epoch_hist.append(int(epoch))
            loss_hist.append(float(loss.item()))
            val_hist.append(float(val_acc))
            test_hist.append(float(test_acc))



        if epoch % 10 == 0:
            print(f"[v3] epoch={epoch:03d} | loss={loss:.5f} | val={val_acc:.4f} | test={test_acc:.4f}")
            torch.save(model.state_dict(), CKPT_PATH)


    torch.save(model.state_dict(), CKPT_PATH)
    print(f"saved checkpoint to {CKPT_PATH}")
    print("checkpoint saved: gnn_facebook.pt")
    #loss function
    fig1, ax1 = plt.subplots()
    ax1.set_title("Training Loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.plot(epoch_hist, loss_hist, label="train_loss")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig1)

    #Accuracy
    fig2, ax2 = plt.subplots()
    ax2.set_title("Validation/Test Accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.plot(epoch_hist, val_hist, label="val_acc")
    ax2.plot(epoch_hist, test_hist, label="test_acc")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(run_dir / "acc_curve.png", dpi=150)
    plt.close(fig2)

if __name__ == "__main__":
    train()
