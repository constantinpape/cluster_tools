#! /usr/bin/python

import os
import z5py
import argparse
import time
import json

import numpy as np
import nifty.distributed as ndist


def features_step1(sub_graph_prefix,
                   data_path, data_key,
                   labels_path, labels_key,
                   offset_file, block_file,
                   out_path):

    t0 = time.time()
    block_list = np.load(block_file).tolist()
    with open(offset_file, 'r') as f:
        offsets = json.load(f)

    dtype = z5py.File(data_path)[data_key].dtype

    if offsets is not None:
        affinity_function = ndist.extractBlockFeaturesFromAffinityMaps_uint8 if dtype == 'uint8' else \
            ndist.extractBlockFeaturesFromAffinityMaps_float32

        affinity_function(sub_graph_prefix,
                          data_path, data_key,
                          labels_path, labels_key,
                          block_list, os.path.join(out_path, 'blocks'),
                          offsets)
    else:
        boundary_function = ndist.extractBlockFeaturesFromBoundaryMaps_uint8 if dtype == 'uint8' else \
            ndist.extractBlockFeaturesFromBoundaryMaps_float32
        boundary_function(sub_graph_prefix,
                          data_path, data_key,
                          labels_path, labels_key,
                          block_list,
                          os.path.join(out_path, 'blocks'))

    job_id = int(os.path.split(block_file)[1].split('_')[2][:-4])
    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sub_graph_prefix", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("data_key", type=str)
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)
    parser.add_argument("--offset_file", type=str)
    parser.add_argument("--block_file", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    features_step1(args.sub_graph_prefix,
                   args.data_path, args.data_key,
                   args.labels_path, args.labels_key,
                   args.offset_file, args.block_file,
                   args.out_path)
