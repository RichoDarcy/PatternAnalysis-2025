import torch
from pathlib import Path
from dataset import build_data, split
from modules import GNN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os, numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("runs", exist_ok=True)

def main():
    #building and spliting graph
    #Move tensors to device
    g = split(build_data(), seed=1).to(device)

    #counting classes froml abled nodes
    keep = g.y.ne(-1)
    n_classes = int(torch.amax(g.y[keep]).item()) + 1 if keep.any() else 0



    model = GNN(g.num_node_features, 128, n_classes, 0.5)
    model.to(device)
    ckpt = str(Path("runs") / "gnn_facebook.pt")
    #checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))

    model.eval()


    #forward pass for predictions without dropout
    with torch.no_grad():
        logits = model(g.x, g.edge_index, use_dropout=False) #logits/activatation per node
        #if trunk returned embeddings only use linear head
        if logits.shape[-1] != n_classes and hasattr(model, "out_linear"):
            logits = model.out_linear(logits)

        preds = logits.argmax(dim=-1)
        test_mask = g.test_mask
        correct = preds.eq(g.y) & test_mask
        acc = correct.sum().item() / int(test_mask.sum().item())
        
    


    
    #labelling for visualisation
    lab = (g.y >= 0)
    idx = lab.nonzero(as_tuple=False).view(-1)

    yt = g.y.index_select(0, idx).cpu().numpy()
    yp = preds.index_select(0, idx).cpu().numpy()
    fl = (yt != yp).astype(np.int32)

    #get stable embeddings
    with torch.no_grad():
        H = model.get_embed(g.x, g.edge_index).cpu().numpy()
    #t - SNE on labelled subest
    # #uses Pca for stability    
    Z = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=0)\
        .fit_transform(H[lab.cpu().numpy()])
    #scatter plots
    views = {
        "tsne_gt.png":   (yt, "tab20", "t-SNE truth"),
        "tsne_pred.png": (yp, "tab20", "t-SNE prediction"),
        "tsne_err.png":  (fl, "gray",  "t-SNE error"),
    }

    for fname, (color, cmap_name, title) in views.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(Z[:, 0], Z[:, 1], c=color, s=3, alpha=0.85, cmap=cmap_name)
        ax.set_title(title)
        fig.tight_layout()
        out = os.path.join("runs", fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved", out)

    print(f"test_acc={acc:.4f}")



    

if __name__ == "__main__":
    main()
