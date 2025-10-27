import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
        obj = json.load(fh)
    if isinstance(obj, dict):
        obj = obj["features"] if ("features" in obj and isinstance(obj["features"], dict)) else obj
        return {str(k): np.asarray(list(map(float, v)), dtype=np.float32) for k, v in obj.items()}
    if isinstance(obj, list):
        return {str(r["id"]): np.asarray(list(map(float, r["features"])), dtype=np.float32)
                for r in obj if isinstance(r, dict) and "id" in r and "features" in r}
    raise ValueError("unsupported features json")
