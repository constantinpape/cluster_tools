#! /usr/bin/python

import argparse
import nifty
import nifty.distributed as ndist
import z5py


def graph_step4(graph_path, scale, initial_block_shape, n_threads):

    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    f_graph = z5py.File(graph_path)
    shape = f_graph.attrs['shape']
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    input_key = 'graph'

    ndist.mapEdgeIdsForAllBlocks(graph_path, input_key,
                                 blockGroup='sub_graphs/s%s' % scale,
                                 blockPrefix="block_",
                                 numberOfBlocks=blocking.numberOfBlocks,
                                 numberOfThreads=n_threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("--initial_block_shape", nargs=3, type=int)
    parser.add_argument("--n_threads", type=int)
    args = parser.parse_args()

    graph_step4(args.graph_path, args.last_scale,
                list(args.initial_block_shape), args.n_threads)
