#! /usr/bin/python

import time
import os
import argparse
import numpy as np

import z5py
import nifty


def blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder, output_prefix):
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_blocks = blocking.numberOfBlocks
    assert n_jobs <= n_blocks, "%i, %i" % (n_jobs, n_blocks)

    # assign blocks to jobs
    block_list = list(range(n_blocks))
    for job_id in range(n_jobs):
        np.save(os.path.join(tmp_folder, '%s_%i.npy' % (output_prefix, job_id)),
                block_list[job_id::n_jobs])
    return n_blocks


def serialize_initial_problem(graph_path, tmp_folder, costs_path, costs_key):

    # make merged - graph n5 file
    graph_out_path = os.path.join(tmp_folder, 'merged_graph.n5')
    f_graph = z5py.File(graph_out_path, use_zarr_format=False)

    # make scale 0 group
    if 's0' not in f_graph:
        f_graph.create_group('s0')

    # TODO fix symlinks
    # make symbolic link to the costs
    os.symlink(os.path.join(costs_path, costs_key), os.path.join(graph_out_path, 's0', 'costs'))

    # FIXME this does not work, for some odd reason
    # make symlinks to the normal, zero-level graph
    # print("Symlink from", os.path.join(graph_path, 'graph'))
    # print("to", os.path.join(graph_out_path, 's0', 'graph'))
    # os.symlink(os.path.join(graph_path, 'graph'), os.path.join(graph_out_path, 's0', 'graph'))


def prepare(graph_path, graph_key,
            costs_path, costs_key,
            initial_block_shape,
            n_scales,
            tmp_folder,
            n_jobs,
            n_threads):

    t0 = time.time()
    # load number of nodes and the shape
    f_graph = z5py.File(graph_path)
    shape = f_graph.attrs['shape']

    # make tmp folder
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    serialize_initial_problem(graph_path, tmp_folder, costs_path, costs_key)

    # block mappings for next steps
    for scale in range(n_scales):
        factor = 2**scale
        scale_shape = [bs*factor for bs in initial_block_shape]
        scale_prefix = '2_input_s%i' % scale
        blocks_to_jobs(shape, scale_shape, n_jobs, tmp_folder, scale_prefix)

    t0 = time.time() - t0
    print("Success")
    print("In %f s" % t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("graph_key", type=str)
    parser.add_argument("costs_path", type=str)
    parser.add_argument("costs_key", type=str)

    parser.add_argument("--initial_block_shape", nargs=3, type=int)
    parser.add_argument("--n_scales", type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--n_threads", type=int)
    parser.add_argument("--use_mc_costs", type=int)

    # TODO make cost options settable

    args = parser.parse_args()

    prepare(args.graph_path, args.graph_key,
            args.costs_path, args.costs_key,
            list(args.initial_block_shape), args.n_scales,
            args.tmp_folder, args.n_jobs, args.n_threads,
            bool(args.use_mc_costs))
