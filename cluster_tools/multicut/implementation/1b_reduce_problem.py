#! /usr/bin/python

import time
import os
import argparse
import numpy as np

import vigra
import z5py
import nifty
import nifty.distributed as ndist


# TODO de-spaghettify !!!
def multicut_step1(graph_path, scale,
                   tmp_folder, n_jobs,
                   initial_block_shape, n_threads,
                   cost_accumulation="sum"):
    t0 = time.time()
    # get the number of nodes and uv-ids at this scale level
    # as well as the initial node labeling
    shape = z5py.File(graph_path).attrs['shape']
    if scale == 0:
        f_graph = z5py.File(os.path.join(graph_path, 'graph'), use_zarr_format=False)
        initial_node_labeling = None
        n_nodes = f_graph.attrs['numberOfNodes']
        uv_ids = f_graph['edges'][:]
    else:
        f_problem = z5py.File(os.path.join(tmp_folder, 'problem.n5/s%i' % scale))
        n_nodes = f_problem.attrs['numberOfNodes']
        uv_ids = f_problem['uvIds'][:]
        initial_node_labeling = f_problem['nodeLabeling'][:]
    n_edges = len(uv_ids)

    # get the costs
    costs = z5py.File(os.path.join(tmp_folder, 'problem.n5/s%i' % scale),
                      use_zarr_format=False)['costs'][:]
    assert len(costs) == n_edges, "%i, %i" (len(costs), n_edges)

    # load the cut-edge ids from the prev. jobs and make merge edge ids
    merge_edges = np.ones(n_edges, dtype='bool')
    # TODO we could parallelize this
    cut_edge_ids = np.concatenate([np.load(os.path.join(tmp_folder,
                                                        '1_output_s%i_%i.npy' % (scale, job_id)))
                                   for job_id in range(n_jobs)])
    cut_edge_ids = np.unique(cut_edge_ids)
    merge_edges[cut_edge_ids] = False

    # merge node pairs with ufd
    ufd = nifty.ufd.ufd(n_nodes)
    merge_pairs = uv_ids[merge_edges]
    ufd.merge(merge_pairs)

    # get the node results and label them consecutively
    node_labeling = ufd.elementLabeling()
    node_labeling, max_new_id, _ = vigra.analysis.relabelConsecutive(node_labeling)
    n_new_nodes = max_new_id + 1

    # get the labeling of initial nodes
    if initial_node_labeling is None:
        new_initial_node_labeling = node_labeling
    else:
        # should this ever become a bottleneck, we can parallelize this in nifty
        # but for now this would really be premature optimization
        new_initial_node_labeling = node_labeling[initial_node_labeling]

    # get new edge costs
    edge_mapping = nifty.tools.EdgeMapping(uv_ids, node_labeling, numberOfThreads=n_threads)
    new_uv_ids = edge_mapping.newUvIds()

    new_costs = edge_mapping.mapEdgeValues(costs, cost_accumulation, numberOfThreads=n_threads)
    assert len(new_uv_ids) == len(new_costs)

    # map the new graph (= node labeling and corresponding edges)
    # to the next scale level

    f_graph = z5py.File(graph_path, use_zarr_format=False)
    f_graph.create_group('merged_graphs/s%i' % scale)
    if scale == 0:
        block_in_prefix = os.path.join(graph_path, 'sub_graphs', 's%i' % scale, 'block_')
    else:
        block_in_prefix = os.path.join(graph_path, 'merged_graphs', 's%i' % scale, 'block_')

    block_out_prefix = os.path.join(graph_path, 'merged_graphs', 's%i' % (scale + 1,), 'block_')

    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    new_factor = 2**(scale + 1)
    new_block_shape = [new_factor * bs for bs in initial_block_shape]

    edge_labeling = edge_mapping.edgeMapping()

    ndist.serializeMergedGraph(block_in_prefix, shape,
                               block_shape, new_block_shape,
                               n_new_nodes,
                               node_labeling, edge_labeling,
                               block_out_prefix, n_threads)

    # serialize all results for the next scale
    problem_out_file = os.path.join(tmp_folder, 'problem.n5')
    f_out = z5py.File(problem_out_file, use_zarr_format=False)

    scale_key = 's%i' % (scale + 1,)
    if scale_key not in f_out:
        f_out.create_group(scale_key)
    g_out = f_out[scale_key]
    g_out.attrs['numberOfNodes'] = n_new_nodes

    n_new_edges = len(new_uv_ids)
    shape_edges = (n_new_edges, 2)
    if 'uvIds' not in g_out:
        ds_edges = g_out.create_dataset('uvIds', dtype='uint64', shape=shape_edges, chunks=shape_edges)
    else:
        ds_edges = g_out['uvIds']
        assert ds_edges.shape == shape_edges
    ds_edges[:] = new_uv_ids

    shape_nodes = (len(new_initial_node_labeling),)
    if 'nodeLabeling' not in g_out:
        ds_nodes = g_out.create_dataset('nodeLabeling', dtype='uint64', shape=shape_nodes, chunks=shape_nodes)
    else:
        ds_nodes = g_out['nodeLabeling']
        assert ds_nodes.shape == shape_nodes
    ds_nodes[:] = new_initial_node_labeling

    shape_costs = (n_new_edges,)
    if 'costs' not in g_out:
        ds_costs = g_out.create_dataset('costs', dtype='float32', shape=shape_costs, chunks=shape_costs)
    else:
        ds_costs = g_out['costs']
    ds_costs[:] = new_costs

    print("Success")
    print("In %f s" % (time.time() - t0,))
    print("Reduced graph from", n_nodes, "to", n_new_nodes, "nodes;",
          n_edges, "to", n_new_edges, "edges.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--initial_block_shape", type=int, nargs=3)
    parser.add_argument("--n_threads", type=int)
    parser.add_argument("--cost_accumulation", type=str)
    args = parser.parse_args()

    multicut_step1(args.graph_path,
                   args.scale, args.tmp_folder,
                   args.n_jobs, args.initial_block_shape,
                   args.n_threads, args.cost_accumulation)
