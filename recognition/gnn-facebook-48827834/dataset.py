import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE = os.path.join(os.path.dirname(__file__), "facebook_large")
F_EDGES = "musae_facebook_edges.csv"
F_TARGET = "musae_facebook_target.csv"
F_FEAT = "musae_facebook_features.json"


def loadedges(path: str):
    #Opens the edge list CSV and pulls out the two columns showing which nodes connect.
    #Figures out whether they’re called source/target or id_1/id_2, then returns two arrays of node IDs.
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    s = cols.get("source", cols.get("id_1"))
    t = cols.get("target", cols.get("id_2"))
    
    return df[s].astype(str).to_numpy(), df[t].astype(str).to_numpy()

def loadtargets(path: str):
    #Loads the node label file, finds the ID column and whichever column holds the class label (target, page_type, or the first available one).
    #Converts those labels into numeric form and returns both the IDs and encoded labels.
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    if "target" in cols:
        raw = df[cols["target"]].to_numpy()
    elif "page_type" in cols:
        raw = df[cols["page_type"]].astype(str).to_numpy()
    else:
        cand = [c for c in df.columns if c != cols["id"]]
        raw = df[cand[0]].astype(str).to_numpy()
    lab = LabelEncoder().fit_transform(raw)
    ids = df[cols["id"]].astype(str).to_numpy()
    return ids, lab