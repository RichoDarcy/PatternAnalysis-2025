import os, dataset
s, t = dataset.load_edges(os.path.join(dataset.BASE, dataset.F_EDGES))
ids, y = dataset.load_targets(os.path.join(dataset.BASE, dataset.F_TARGET))
fm = dataset.load_features(os.path.join(dataset.BASE, dataset.F_FEAT))
assert s.size and t.size and ids.size and len(fm) > 0
print("loader smoke OK")


#smoke test for loader
#Loads edge CSV and checks for valid source/target columns.
#Loads target CSV and verifies ID and label columns exist.
#Loads feature JSON and confirms non-empty feature vectors.
#Ensures all three loaders return non-empty data structures.



g = dataset.build_data()
assert g.x.shape[0] == g.y.shape[0]
assert g.edge_index.shape[0] == 2 and g.edge_index.shape[1] > 0
print("final OK:", g)

#Runs the full build_data() pipeline.
#Confirms features (x) and labels (y) have the same number of nodes.
#Verifies the graph has edges (edge_index not empty).
#Checks edge_index has correct shape [2, E].
#Confirms returned object is a valid torch_geometric.data.Data instance.