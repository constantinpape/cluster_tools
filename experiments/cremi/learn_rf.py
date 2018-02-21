import os
import pickle

import numpy as np
import z5py
import nifty.graph.rag as nrag
from sklearn.ensemble import RandomForestClassifier


def get_labels(path, n_threads=20):
    ws = z5py.File(path)['segmentations/watershed'][:]
    rag = nrag.gridRag(ws,
                       numberOfLabels=int(max(ws)) + 1,
                       numberOfThreads=n_threads)
    uvs = rag.uvIds()
    valid_edges = np.logical_not((uvs == 0).any(axis=1))
    gt = z5py.File(path)['segmentations/groundtruth'][:]
    labels = nrag.accumulateLabels(rag, gt)
    assert labels.shape == valid_edges.shape
    return labels, valid_edges


def learn_rf(out_path):
    samples = ['A', 'B', 'C']

    all_features = []
    all_labels = []

    for sample in samples:
        cache_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/cremi_A/tmp_files' % sample
        path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
        # TODO
        features = z5py.File(os.path.join(cache_folder, ''))['features'][:]
        labels, valid_edges = get_labels(path)
        assert len(features) == len(labels)
        features = features[valid_edges]
        labels = labels[valid_edges]

        all_features.append(features)
        all_labels.append(labels)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    rf = RandomForestClassifier(n_jobs=40)
    rf.fit(features, labels)
    with open(out_path, 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    path = ''
    learn_rf(path)
