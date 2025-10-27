import os, dataset
s, t = dataset.load_edges(os.path.join(dataset.BASE, dataset.F_EDGES))
ids, y = dataset.load_targets(os.path.join(dataset.BASE, dataset.F_TARGET))
fm = dataset.load_features(os.path.join(dataset.BASE, dataset.F_FEAT))
assert s.size and t.size and ids.size and len(fm) > 0
print("loader smoke OK")


#smoke test for loader