import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    #Two layer GNN: 
    #GCN → ReLU → GCN → ReLU → linear head)
    #returns logits per node at inference/training time.
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        #Constructs the model modules: 
        #first and second GCNConv layers
        #a final nn.Linear classifier
        # And stores the dropout probability
        nn.Module.__init__(self)

        self.conv_first  = GCNConv(in_dim, hidden)

        self.conv_second = GCNConv(hidden, hidden)
        self.out_linear  = nn.Linear(hidden, out_dim)
        self.drop_prob   = float(dropout)

    def forward(self, x, edge_index, use_dropout=True):
        #Internal helper that runs the two GCN layers with ReLU over input features feats and edge_index edges
        #Returns the final hidden features tensor of shape [num_nodes, hidden].

        h = self.stackflow(x, edge_index, use_dropout)
        return self.out_linear(h)

 
    def get_embed(self, feats, edges):
        
        return self._flow(feats, edges, use_dropout=False)


    def stackflow(self, x, edge_index, use_dropout: bool):
    
        
        h = self.conv_first(x, edge_index)
        h = F.relu(h, inplace=False)
        if use_dropout:
            h = F.dropout(h, p=self.drop_prob, training=self.training)
        h = self.conv_second(h, edge_index)
        h = F.relu(h, inplace=False)
        if use_dropout:
            h = F.dropout(h, p=self.drop_prob, training=self.training)
        return h
   


    def get_embed(self, x, edge_index):
        #Inference-only pathway 
        #Returns stable node embeddings after the second GCN layer, shape [num_nodes, hidden].
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                return self.stackflow(x, edge_index, use_dropout=False)
        finally:
            self.train(was_training)