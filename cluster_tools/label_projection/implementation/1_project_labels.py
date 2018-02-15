#! /usr/bin/python

import time
import os
import argparse
import numpy as np

import z5py
import nifty.distributed as ndist


def label_projection_step1(labels_path, labels_key,
                           out_path, out_key,
                           node_labeling_path, node_labeling_key,
                           block_shape, block_file):
    t0 = time.time()
    node_labeling = z5py.File(node_labeling_path)[node_labeling_key][:]
    block_ids = np.load(block_file)

    ndist.nodeLabelingToPixels(os.path.join(labels_path, labels_key),
                               os.path.join(out_path, out_key),
                               node_labeling, block_ids,
                               block_shape)

    job_id = int(os.path.split(block_file)[1].split('_')[2][:-4])
    # the zeroth job writes the max-id to the out dataset attributes
    if job_id == 0:
        max_id = int(node_labeling.max())
        z5py.File(out_path)[out_key].attrs['maxId'] = max_id

    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("node_labeling_path", type=str)
    parser.add_argument("node_labeling_key", type=str)
    parser.add_argument("--block_shape", type=int, nargs=3)
    parser.add_argument("--block_file", type=str)
    args = parser.parse_args()

    label_projection_step1(args.labels_path, args.labels_key,
                           args.out_path, args.out_key,
                           args.node_labeling_path, args.node_labeling_key,
                           args.block_shape, args.block_file)
