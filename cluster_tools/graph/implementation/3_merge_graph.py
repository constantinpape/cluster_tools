#! /usr/bin/python

import argparse
import time

import z5py
import nifty
import nifty.distributed as ndist


def graph_step3(graph_path, last_scale, initial_block_shape, n_threads):

    t0 = time.time()
    factor = 2**last_scale
    block_shape = [factor * bs for bs in initial_block_shape]

    f_graph = z5py.File(graph_path)
    shape = f_graph.attrs['shape']
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)

    block_prefix = 'sub_graphs/s%i/block_' % last_scale
    output_key = 'graph'
    block_list = list(range(blocking.numberOfBlocks))
    ndist.mergeSubgraphs(graph_path,
                         blockPrefix=block_prefix,
                         blockIds=block_list,
                         outKey=output_key,
                         numberOfThreads=n_threads)
    f_graph[output_key].attrs['shape'] = shape
    print("Success")
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("last_scale", type=int)
    parser.add_argument("--initial_block_shape", nargs=3, type=int)
    parser.add_argument("--n_threads", type=int)
    args = parser.parse_args()

    graph_step3(args.graph_path, args.last_scale,
                list(args.initial_block_shape), args.n_threads)
