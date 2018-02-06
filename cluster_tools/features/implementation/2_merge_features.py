#! /usr/bin/python

import os
import argparse
import time
import numpy as np
import nifty.distributed as ndist


def features_step2(graph_block_prefix, features_tmp_prefix,
                   out_path, out_key,
                   input_file, n_threads):
    t0 = time.time()
    edge_begin, edge_end, n_blocks = np.load(input_file)
    ndist.mergeFeatureBlocks(graph_block_prefix,
                             features_tmp_prefix,
                             os.path.join(out_path, out_key),
                             n_blocks, edge_begin, edge_end,
                             numberOfThreads=n_threads)

    job_id = int(os.path.split(input_file)[1].split('_')[2][:-4])
    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_block_prefix", type=str)
    parser.add_argument("features_tmp_prefix", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("input_file", type=str)
    parser.add_argument("n_threads", type=str)
    args = parser.parse_args()

    features_step2(args.graph_block_prefix, args.features_tmp_prefix,
                   args.out_path, args.out_key,
                   args.input_file, args.n_threads)
