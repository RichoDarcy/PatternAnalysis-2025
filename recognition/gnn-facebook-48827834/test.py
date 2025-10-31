# from dataset import *


# import os
# import json
# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import to_undirected
# from sklearn.preprocessing import LabelEncoder, QuantileTransformer, normalize
# from modules import *
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# import torch
# import torch.nn.functional as F
# from dataset import build_data, split
# from modules import GNN
# from train import *
# import os
# import csv
# from pathlib import Path
# import matplotlib.pyplot as plt


# #unify index

# f = {"a": torch.ones(2), "b": torch.zeros(2)}
# y = np.array(["b","c","c"])
# s = np.array(["a","x"])
# t = np.array(["b","a"])
# n,d = unify_index(f,y,s,t)
# # 
# print( len(n), sorted(list(d.keys())))

# assert "a" in d and "b" in d and "c" in d
# assert isinstance(n, np.ndarray)


# #assemble_features
# f = {"a": np.array([1.,2.],dtype=np.float32),"b": np.array([9.],dtype=np.float32)}
# n = np.array(["a","b","c"],dtype=object)
# d = {"a":0,"b":1,"c":2}
# x = assemble_features(f,n,d)

# print(x.shape, x[0].tolist(), x[1].tolist())
# assert x.shape==(3,2)
# assert float(x[0,0])==1.0 and float(x[0,1])==2.0
# assert float(x[1,0])==9.0 and float(x[1,1])==0.0


# #scale  features
# x0 = np.array([[0.,1.,2.],[3.,4.,5.]],dtype=np.float32)
# t0 = scale_features(x0)
# print(type(t0).__name__,tuple(t0.shape))

# assert isinstance(t0, torch.Tensor)
# assert t0.shape==torch.Size([2,3])



# #maplabels
# i = np.array(["u","v"])
# y = np.array([3,1])
# n = np.array(["v","w","u"],dtype=object)
# d = {"v":0,"w":1,"u":2}

# y2 = map_labels(i,y,n,d)

# print(y2.tolist())
# assert y2.tolist()==[1,-1,3]



# #build edges
# s = np.array(["a","a","x","b"])
# t = np.array(["b","a","y","b"])
# d = {"a":0,"b":1}

# e = build_edges(s,t,d).t().tolist()
# es = {tuple(q) for q in e}

# print(sorted(list(es)))

# assert (0,1) in es and (1,0) in es
# assert (0,0) not in es and (1,1) not in es
# assert len(es)==2



# #split
# x = torch.randn(30,4); y = torch.randint(0,3,(30,))
# e = torch.tensor([list(range(29)),list(range(1,30))])

# g = Data(x=x,y=y,edge_index=e)
# g = split(g,seed=9)
# n = x.shape[0]

# a = int(g.train_mask.sum().item())
# b = int(g.val_mask.sum().item())
# c = int(g.test_mask.sum().item())
# print( a,b,c,n)

# assert a+b+c==n and a>0 and b>0 and c>0




# #forward
# #!! MLP
# x = torch.randn(5, 7)
# edge_index = torch.zeros(2, 0, dtype=torch.long)

# m = MLP(in_dim=7, hidden=8, out_dim=3, dropout=0.5).eval()
# logits = m(x, edge_index=edge_index)

# print(tuple(logits.shape))
# assert logits.shape == (5, 3)






# # get_embeddings
# #!!!mlp
# m.train()
# emb = m.get_embeddings(x, edge_index=None)

# print(tuple(emb.shape))
# assert emb.shape[0] == 5 and emb.shape[1] == 8




# #forward

# x = torch.randn(6,5)
# e = torch.tensor([[0,1,2,3,4],[1,2,3,4,5]])
# m = GNN(5,7,3,0.25)
# z = m(x,e)

# print(tuple(z.shape))
# assert tuple(z.shape)==(6,3)






# #get_embed
# u = m.get_embed(x,e)
# print(tuple(u.shape))

# assert u.shape[0]==x.shape[0]
# assert u.dim()==2






# #accuracy

# L = torch.tensor([[0.2,0.8],[0.9,0.1],[0.3,0.7],[0.6,0.4]])
# y = torch.tensor([1,0,1,0])
# a = accuracy(L,y)
# print(float(a))

# assert 0.0 <= a <= 1.0
# assert abs(a - 1.0) < 1e-6




# #num_classes
# y = torch.tensor([3,2,1,-1,0,3,-1])
# c = num_classes(y,-1)

# print(int(c))
# assert c==4







# #infer
# g = Data(x=x, edge_index=e)
# o = infer(m,g,False,3)

# print(tuple(o.shape))

# assert o.shape[0]==x.shape[0]
# assert o.shape[1]==3
