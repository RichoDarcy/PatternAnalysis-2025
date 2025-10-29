import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, normalize

BASE = os.path.join(os.path.dirname(__file__), "facebook_large")
F_EDGES = "musae_facebook_edges.csv"
F_TARGET = "musae_facebook_target.csv"
F_FEAT = "musae_facebook_features.json"

def load_edges(path: str):
    #Opens the edge list CSV and pulls out the two columns showing which nodes connect.
    #Figures out whether they’re called source/target or id_1/id_2, then returns two arrays of node IDs.
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    s = cols.get("source", cols.get("id_1"))
    t = cols.get("target", cols.get("id_2"))
    if s is None or t is None:
        raise ValueError("edges csv requires source/target (or id_1/id_2)")
    return df[s].astype(str).to_numpy(), df[t].astype(str).to_numpy()

def load_targets(path: str):
    #Loads the node label file, finds the ID column and whichever column holds the class label (target, page_type, or the first available one).
    #Converts those labels into numeric form and returns both the IDs and encoded labels.
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "id" not in cols:
        raise ValueError("targets csv requires id column")
    if "target" in cols:
        raw = df[cols["target"]].to_numpy()
    elif "page_type" in cols:
        raw = df[cols["page_type"]].astype(str).to_numpy()
    else:
        cand = [c for c in df.columns if c != cols["id"]]
        if not cand:
            raise ValueError("no label column")
        raw = df[cand[0]].astype(str).to_numpy()
    lab = LabelEncoder().fit_transform(raw)
    ids = df[cols["id"]].astype(str).to_numpy()
    return ids, lab


def load_features(path: str):
    #Reads a JSON file containing node features.
    #Handles both dictionary and list-based formats, then builds a mapping from node ID to its feature vector as a NumPy array of floats.
    with open(path, "r", encoding="utf-8") as fh:
        try:
            obj = json.load(fh)
        except json.JSONDecodeError:
            fh.seek(0)
            obj = [json.loads(line) for line in fh if line.strip()]
    if isinstance(obj, dict):
        obj = obj["features"] if ("features" in obj and isinstance(obj["features"], dict)) else obj
        return {str(k): np.asarray(list(map(float, v)), dtype=np.float32) for k, v in obj.items()}
    if isinstance(obj, list):
        return {str(r["id"]): np.asarray(list(map(float, r["features"])), dtype=np.float32)
                for r in obj if isinstance(r, dict) and "id" in r and "features" in r}
    raise ValueError("unsupported features json")


def unify_index(feat_map, y_ids, s_arr, t_arr):
    #Collects every node ID that appears anywhere in features targets or edges.
    #It builds a master list of unique nodes along with a dictionary mapping each ID to its numeric index.
    ordered = list(feat_map.keys()) + list(y_ids) + list(pd.unique(s_arr)) + list(pd.unique(t_arr))
    nodes = np.array(list(dict.fromkeys(ordered)), dtype=object)
    id2ix = {k: i for i, k in enumerate(nodes)}
    return nodes, id2ix



def assemble_features(feat_map, nodes, id2ix):
    #Creates a complete feature matrix for all nodes.
    #If some nodes have shorter vectors, it pads them with zeros; if longer, it trims them. 
    #Returns the resulting NumPy array.
    d = max((len(v) for v in feat_map.values()), default=0)
    if d == 0:
        raise ValueError("no features found")
    X = np.zeros((nodes.size, d), dtype=np.float32)
    for k, v in feat_map.items():
        i = id2ix[k]
        if v.size >= d:
            X[i] = v[:d]
        else:
            row = np.zeros(d, dtype=np.float32)
            row[:v.size] = v
            X[i] = row
    return X



def scale_features(X: np.ndarray):
    #Normalizes the feature matrix. It first transforms its values to follow a roughly normal distribution
    #then L2-normalizing each row and turns it into a PyTorch tensor
    n_q = min(100, max(10, X.shape[0] // 10))
    n_q = min(n_q, X.shape[0])
    qt = QuantileTransformer(n_quantiles=n_q, output_distribution="normal", subsample=int(1e9))
    X = qt.fit_transform(X)
    X = normalize(X, norm="l2", axis=1, copy=False)
    return torch.tensor(X, dtype=torch.float32)



def map_labels(y_ids, y_lab, nodes, id2ix):
    #Places every known label into the right position in a full label vector aligned with the node order
    #Nodes without labels stay as zeros
    # Returns a PyTorch tensor.
    y = np.full(nodes.size, fill_value=-1, dtype=np.int64)
    pos = np.fromiter((id2ix.get(k, -1) for k in y_ids), dtype=np.int64)
    keep = pos >= 0
    if keep.any():
        y[pos[keep]] = y_lab[keep]

    return torch.from_numpy(y)




def build_edges(s_arr, t_arr, id2ix):
    #Converts all source and target IDs into integer node indices
    #removes any self-connections or missing IDs, and returns a symmetric edge index tensor.
    sx = np.fromiter((id2ix.get(k, -1) for k in s_arr), dtype=np.int64)
    tx = np.fromiter((id2ix.get(k, -1) for k in t_arr), dtype=np.int64)
    keep = (sx >= 0) & (tx >= 0) & (sx != tx)
    E = np.vstack([sx[keep], tx[keep]]).astype(np.int64)
    E = torch.tensor(E, dtype=torch.long)
    return to_undirected(E)




def build_data(root: str = BASE):
    #Runs the entire loading process
    #reads in edges, targets, and features
    #builds the node index
    #assembles and scales load_feature
    #maps labels; builds edges
    #returns a complete torch_geometric.data.Data object ready for use.
    s_arr, t_arr = load_edges(os.path.join(root, F_EDGES))
    y_ids, y_lab = load_targets(os.path.join(root, F_TARGET))
    feat_map = load_features(os.path.join(root, F_FEAT))
    nodes, id2ix = unify_index(feat_map, y_ids, s_arr, t_arr)
    X_np = assemble_features(feat_map, nodes, id2ix)
    X = scale_features(X_np)
    y = map_labels(y_ids, y_lab, nodes, id2ix)
    E = build_edges(s_arr, t_arr, id2ix)
    return Data(x=X, y=y, edge_index=E)


def split(g, train_frac=0.6, val_frac=0.2, seed=1, unlabeled=-1):
    #shuffles labeled nodes and attach boolean masks for a 60/20/20 train/val/test split
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(g.y.cpu().numpy() != unlabeled)
    idx = rng.permutation(idx)
    n = len(idx)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    g.train_mask = torch.zeros(g.num_nodes, dtype=torch.bool); g.train_mask[tr] = True
    g.val_mask   = torch.zeros(g.num_nodes, dtype=torch.bool); g.val_mask[va] = True
    g.test_mask  = torch.zeros(g.num_nodes, dtype=torch.bool); g.test_mask[te] = True
    return g