#! /usr/bin/python

import time
import os
import argparse
import numpy as np

import z5py
import nifty
import cremi_tools.segmentation as cseg
import nifty.distributed as ndist


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


# TODO multi-threaded implementation
def make_costs(features_path, features_key, ds_graph, tmp_folder,
               costs_for_multicut, beta, weighting_exponent,
               weight_edges, n_threads, invert_inputs=True):

    # find the ignore edges
    uv_ids = ds_graph['edges'][:]
    ignore_edges = (uv_ids == 0).any(axis=1)

    # get the multicut edge costs from mean affinities
    feature_ds = z5py.File(features_path)[features_key]
    if invert_inputs:
        costs = 1. - feature_ds[:, 0:1].squeeze()
    else:
        costs = feature_ds[:, 0:1].squeeze()

    if weight_edges:
        edge_sizes = feature_ds[:, 9:].squeeze()
        # set edge sizes of ignore edges to 1 (we don't want them to influence the weighting)
        edge_sizes[ignore_edges] = 1
    else:
        edge_sizes = None

    if costs_for_multicut:
        costs = cseg.transform_probabilities_to_costs(costs, beta=beta,
                                                      edge_sizes=edge_sizes,
                                                      weighting_exponent=weighting_exponent)

    # set weights of ignore edges to be maximally repulsive
    if costs_for_multicut:
        costs[ignore_edges] = 5 * costs.min()
    else:
        costs[ignore_edges] = 1.

    # write the costs
    costs_out_path = os.path.join(tmp_folder, 'problem.n5')
    f_costs = z5py.File(costs_out_path, use_zarr_format=False)
    cost_shape = (len(costs),)
    if 's0' not in f_costs:
        g = f_costs.create_group('s0')
    else:
        g = f_costs['s0']

    if 'costs' not in g:
        ds = g.create_dataset('costs', dtype='float32', shape=cost_shape, chunks=cost_shape)
    else:
        ds = g['costs']
        assert ds.shape == cost_shape
    ds[:] = costs.astype('float32')


def prepare(graph_path, graph_key,
            features_path, features_key,
            initial_block_shape,
            n_scales,
            tmp_folder,
            n_jobs,
            n_threads,
            costs_for_multicut=True,
            beta=0.5,
            weight_edges=True,
            weighting_exponent=1.):

    t0 = time.time()
    assert os.path.exists(features_path), features_path

    # load number of nodes and the shape
    t1 = time.time()
    f_graph = z5py.File(graph_path)
    shape = f_graph.attrs['shape']
    ds_graph = f_graph[graph_key]
    n_nodes = ds_graph.attrs['numberOfNodes']
    print("Loading graph in", time.time() - t1)

    # make tmp folder
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    # make edge costs
    t2 = time.time()
    make_costs(features_path, features_key, ds_graph, tmp_folder,
               costs_for_multicut, beta, weighting_exponent,
               weight_edges, n_threads)
    print("Making costs in", time.time() - t2)

    t3 = time.time()
    # block mappings for next steps
    for scale in range(n_scales):
        factor = 2**scale
        scale_shape = [bs*factor for bs in initial_block_shape]
        scale_prefix = '2_input_s%i' % scale
        blocks_to_jobs(shape, scale_shape, n_jobs, tmp_folder, scale_prefix)
    print("Making jobs in", time.time() - t3)

    # get node to block assignment for scale level 0 and the oversegmentaton nodes
    node_out = os.path.join(tmp_folder, 'nodes_to_blocks')

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=initial_block_shape)
    n_initial_blocks = blocking.numberOfBlocks

    if not os.path.exists(node_out):
        os.mkdir(node_out)
    t4 = time.time()
    ndist.nodesToBlocks(os.path.join(graph_path, 'sub_graphs/s0/block_'),
                        os.path.join(node_out, 's0.h5'),
                        numberOfBlocks=n_initial_blocks,
                        numberOfNodes=n_nodes,
                        numberOfThreads=n_threads)
    print("Making nodes to blocks in", time.time() - t4)

    t0 = time.time() - t0
    print("Success")
    print("In %f s" % t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("graph_key", type=str)
    parser.add_argument("features_path", type=str)
    parser.add_argument("features_key", type=str)

    parser.add_argument("--initial_block_shape", nargs=3, type=int)
    parser.add_argument("--n_scales", type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--n_threads", type=int)
    parser.add_argument("--use_mc_costs", type=int)

    # TODO make cost options settable

    args = parser.parse_args()

    prepare(args.graph_path, args.graph_key,
            args.features_path, args.features_key,
            list(args.initial_block_shape), args.n_scales,
            args.tmp_folder, args.n_jobs, args.n_threads,
            bool(args.use_mc_costs))
