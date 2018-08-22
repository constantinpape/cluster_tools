import os
import pickle
import numpy as np

import nifty.graph.rag as nrag
# import nifty.graph.opt.lifted_multicut as nlmc

import z5py
from sklearn.ensemble import RandomForestClassifier

from .. import features as feat


def extract_feats_and_labels(path, aff_key, ws_key, gt_key, mask_key,
                             n_threads=40, learn_2_rfs=True, with_glia=False):
    f = z5py.File(path)

    # load the watershed segmentation and compute rag
    ds_seg = f[ws_key]
    ds_seg.n_threads = n_threads
    seg = ds_seg[:]
    n_labels = int(seg.max()) + 1
    rag = nrag.gridRag(seg, numberOfLabels=n_labels,
                       numberOfThreads=n_threads)
    uv_ids = rag.uvIds()

    # load affinities and glia channel
    ds_affs = f[aff_key]
    ds_affs.n_threads = n_threads
    affs = ds_affs[:3]
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = 1. - affs

    # TODO enable splitting xy and z features
    # get the edge features
    features, _, z_edges = feat.edge_features(rag, seg, n_labels, uv_ids, affs,
                                              n_threads=n_threads)

    # glia features
    if with_glia:
        print("Computing glia features")
        n_chans = ds_affs.shape[0]
        glia_slice = slice(n_chans - 1, n_chans)
        glia = ds_affs[glia_slice]
        if glia.dtype == np.dtype('uint8'):
            glia = glia.astype('float32') / 255.
        np.concatenate([features,
                        feat.region_features(seg, uv_ids, glia)], axis=1)

    # load mask and groundtruth
    ds_mask = f[mask_key]
    ds_mask.n_threads = n_threads
    mask = ds_mask[:]

    ds_gt = f[gt_key]
    ds_gt.n_threads = n_threads
    gt = ds_gt[:]
    gt[np.logical_not(mask)] = 0

    # compute the edge labels and valid edges
    node_labels = nrag.gridRagAccumulateLabels(rag, gt)
    labels = (node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]).astype('uint8')
    valid_edges = (node_labels[uv_ids] != 0).all(axis=1)
    print(np.sum(valid_edges), "edges of", len(uv_ids), "are valid")
    assert features.shape[0] == labels.shape[0]

    # just for temporary inspection, deactivate !
    import vigra
    vigra.writeHDF5(features, './feats_tmp.h5', 'data', chunks=True)
    vigra.writeHDF5(labels, './labs_tmp.h5', 'data', chunks=True)

    if learn_2_rfs:
        features = features[valid_edges]
        labels = labels[valid_edges]
        z_edges = z_edges[valid_edges]
        return (features[np.logical_not(z_edges)], features[z_edges],
                labels[np.logical_not(z_edges)], labels[z_edges])

    else:
        return features[valid_edges], labels[valid_edges]


# TODO enable learning with 2 random forests
def learn_local_rfs(paths, save_path,
                    aff_key='volumes/predictions/affinities',
                    ws_key='volumes/labels/watershed_2d',
                    gt_key='volumes/labels/neuron_ids',
                    mask_key='volumes/labels/mask',
                    n_threads=40, n_trees=200,
                    max_depth=None, with_glia=False):
    assert all(os.path.exists(path) for path in paths)
    features_xy, labels_xy = [], []
    features_z, labels_z = [], []
    for path in paths:
        print("Computing features and labels from", path)
        feats_xy, feats_z, labs_xy, labs_z = extract_feats_and_labels(path, aff_key, ws_key,
                                                                      gt_key, mask_key,
                                                                      n_threads=n_threads,
                                                                      learn_2_rfs=True,
                                                                      with_glia=with_glia)
        features_xy.append(feats_xy)
        labels_xy.append(labs_xy)
        features_z.append(feats_z)
        labels_z.append(labs_z)
    features_xy = np.concatenate(features_xy, axis=0)
    labels_xy = np.concatenate(labels_xy, axis=0)
    features_z = np.concatenate(features_z, axis=0)
    labels_z = np.concatenate(labels_z, axis=0)
    assert len(features_xy) == len(labels_xy)
    assert len(features_z) == len(labels_z)

    print("Start fitting rf xy ...")
    rf_xy = RandomForestClassifier(n_jobs=n_threads, n_estimators=n_trees,
                                   class_weight='balanced', max_depth=max_depth)
    rf_xy.fit(features_xy, labels_xy)
    rf_xy.n_jobs = 1
    print("... done")
    path_xy = save_path.split('.')[0] + '_xy.pkl'
    with open(path_xy, 'wb') as f:
        pickle.dump(rf_xy, f)

    print("Start fitting rf z ...")
    rf_z = RandomForestClassifier(n_jobs=n_threads, n_estimators=n_trees,
                                  class_weight='balanced', max_depth=max_depth)
    rf_z.fit(features_z, labels_z)
    print("... done")
    path_z = save_path.split('.')[0] + '_z.pkl'
    with open(path_z, 'wb') as f:
        pickle.dump(rf_z, f)
