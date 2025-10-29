import torch
from pathlib import Path
from dataset import build_data, split
from modules import GNN

def main():
    g = split(build_data(), seed=1)


    keep = g.y.ne(-1)
    n_classes = int(torch.amax(g.y[keep]).item()) + 1 if keep.any() else 0



    model = GNN(g.num_node_features, 128, n_classes, 0.5)
    ckpt = str(Path("runs") / "gnn_facebook.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()


    with torch.no_grad():
        logits = model(g.x, g.edge_index, use_dropout=False)
        if logits.shape[-1] != n_classes and hasattr(model, "out_linear"):
            logits = model.out_linear(logits)

        preds = logits.argmax(dim=-1)
        test_mask = g.test_mask
        correct = preds.eq(g.y) & test_mask
        acc = correct.sum().item() / int(test_mask.sum().item())
        

    print(f"test_acc={acc:.4f}")

if __name__ == "__main__":
    main()
