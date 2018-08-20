#! /usr/bin/python

import os
import time
import argparse
import numpy as np

import z5py
import nifty
import nifty.distributed as ndist


def merge_subblocks(block_id, blocking, previous_blocking, graph_path, scale):
    block = blocking.getBlock(block_id)
    input_key = 'sub_graphs/s%i/block_' % (scale - 1,)
    output_key = 'sub_graphs/s%i/block_%i' % (scale, block_id)
    block_list = previous_blocking.getBlockIdsInBoundingBox(roiBegin=block.begin,
                                                            roiEnd=block.end,
                                                            blockHalo=[0, 0, 0])
    ndist.mergeSubgraphs(graph_path,
                         blockPrefix=input_key,
                         blockIds=block_list.tolist(),
                         outKey=output_key)


def graph_step2(graph_path, scale, block_file, initial_block_shape):

    t0 = time.time()
    factor = 2**scale
    previous_factor = 2**(scale - 1)
    block_shape = [factor * bs for bs in initial_block_shape]
    previous_block_shape = [previous_factor * bs for bs in initial_block_shape]

    f_graph = z5py.File(graph_path)
    shape = f_graph.attrs['shape']
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    previous_blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                             roiEnd=list(shape),
                                             blockShape=previous_block_shape)
    block_list = np.load(block_file)
    for block_id in block_list:
        merge_subblocks(block_id, blocking, previous_blocking, graph_path, scale)

    job_id = int(os.path.split(block_file)[1].split('_')[3][:-4])
    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("--block_file", type=str)
    parser.add_argument("--initial_block_shape", nargs=3, type=int)
    args = parser.parse_args()

    graph_step2(args.graph_path, args.scale,
                args.block_file, list(args.initial_block_shape))
