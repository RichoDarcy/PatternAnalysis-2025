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

    def forward(self, feats, edges, use_dropout):
        #Internal helper that runs the two GCN layers with ReLU over input features feats and edge_index edges
        #Returns the final hidden features tensor of shape [num_nodes, hidden].

        convs = (self.conv_first, self.conv_second)
        for conv in convs:
            feats = conv(feats, edges)
            feats = F.relu(feats, inplace=False)

            if use_dropout:
                feats = F.dropout(feats, p=self.drop_prob, training=True)
        return feats

 
    def get_embed(self, feats, edges):
        #Inference-only pathway 
        #Returns stable node embeddings after the second GCN layer, shape [num_nodes, hidden].
        return self._flow(feats, edges, use_dropout=False)

   
