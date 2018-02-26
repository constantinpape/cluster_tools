import os
import pickle

import numpy as np
import z5py
import nifty.graph.rag as nrag
from sklearn.ensemble import RandomForestClassifier


def get_labels(path, n_threads=20):
    print("Loading watershed")
    ws = z5py.File(path)['segmentations/watershed'][:]
    print("Computing Rag")
    rag = nrag.gridRag(ws,
                       numberOfLabels=int(ws.max()) + 1,
                       numberOfThreads=n_threads)
    uvs = rag.uvIds()
    valid_edges = np.logical_not((uvs == 0).any(axis=1))
    print("Loading groundtruth")
    gt = z5py.File(path)['segmentations/groundtruth'][:]
    print("Accumulating labels")
    node_labels = nrag.gridRagAccumulateLabels(rag, gt)
    labels = (node_labels[uvs[:, 0]] != node_labels[uvs[:, 1]]).view('uint8')
    assert labels.shape == valid_edges.shape, "%s, %s" % (str(labels.shape), str(valid_edges.shape))
    return labels, valid_edges


def learn_rf(out_path, n_trees=150, n_threads=20):
    samples = ['A', 'B', 'C']
    # samples = ['A']

    all_features = []
    all_labels = []

    for sample in samples:
        print("Getting features and labels for sample", sample)
        cache_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/cremi_%s/tmp_files' % sample
        path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
        features = z5py.File(os.path.join(cache_folder, 'features.n5'))['features'][:]

        # sanity check the featues
        for ii in range(features.shape[1]):
            print("Zero features in column", ii)
            print(np.sum(np.isclose(features[:, ii], 0)))

        labels, valid_edges = get_labels(path)
        assert len(features) == len(labels)
        features = features[valid_edges]
        labels = labels[valid_edges]

        all_features.append(features)
        all_labels.append(labels)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    assert len(features) == len(labels)

    print("Fitting random forest for", len(features), "examples")
    rf = RandomForestClassifier(n_jobs=n_threads, n_estimators=n_trees)
    rf.fit(features, labels)
    with open(out_path, 'wb') as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    path = './rf_ABC.pkl'
    learn_rf(path)
