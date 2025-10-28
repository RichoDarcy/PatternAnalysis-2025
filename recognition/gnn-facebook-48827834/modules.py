import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    #Two-layer feed-forward classifier, hidden layer + output layer for node features
    #ignores graph edges.
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        #Builds the network: a projection Linear(in_dim→hidden)
        # n output head Linear(hidden→out_dim)
        #Stores the dropout rate.
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.p = float(dropout)

    def get_embeddings(self, x, edge_index=None):
        #Runs a forward pass up to the hidden layer ReLU + dropout off.
        #Returns those hidden embeddings for each node.
        h = self.proj(x)
        h = F.relu(h, inplace=False)
        h = F.dropout(h, p=self.p, training=False)
        return h

    def forward(self, x, edge_index=None):
        # Full forward pass: hidden layer using ReLU + dropout honoring training mode
        # followed by the output layer to produce class logits
        h = self.proj(x)
        h = F.relu(h, inplace=False)
        h = F.dropout(h, p=self.p, training=self.training)
        return self.head(h)
    

