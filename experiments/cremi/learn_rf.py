import os
import z5py
import nifty.graph.rag as nrag
from sklearn.ensemble import RandomForestClassifier


def learn_rf():
    samples = ['A', 'B', 'C']

    for sample in samples:
        cache_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/cremi_A/tmp_files' % sample
        path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
        features = z5py.File(os.path.join(cache_folder))
        edges =
        ignore_edges = (edges == 0).any(axis=1)
