#! /usr/bin/python

import os
import time
import pickle
import argparse
import numpy as np
import z5py


def costs_step1(features_path, features_key,
                out_path, out_key,
                rf_path, input_file,
                n_threads_rf):

    t0 = time.time()
    with open(rf_path, 'rb') as f:
        rf = pickle.load(f)
    rf.n_jobs = n_threads_rf

    edge_id_begin, edge_id_end = np.load(input_file)

    feat_roi = np.s_[edge_id_begin:edge_id_end, :]
    feats = z5py.File(features_path)[features_key][feat_roi]

    probs = rf.predict_proba(feats)[:, 1]

    ds_out = z5py.File(out_path)[out_key]
    roi_out = np.s_[edge_id_begin:edge_id_end]
    ds_out[roi_out] = probs.astype('float32', copy=False)

    job_id = int(os.path.split(input_file)[1].split('_')[2][:-4])
    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('features_path', type=str)
    parser.add_argument('features_key', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('out_key', type=str)

    parser.add_argument('rf_path', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--n_threads_rf', type=int)

    args = parser.parse_args()
    costs_step1(args.features_path, args.features_key,
                args.out_path, args.out_key,
                args.rf_path, args.input_file,
                args.n_threads_rf)
