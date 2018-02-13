#! /usr/bin/python

import time
import os
import argparse
import numpy as np

import z5py
import nifty
import cremi_tools.segmentation as cseg
import nifty.distributed as ndist

# TODO support more agglomerators
AGGLOMERATORS = {"multicut_kl": cseg.Multicut("kernighan-lin")}


def solve_block_subproblem(block_id, block_prefix, node_storage_prefix, costs, agglomerator):
    # load the nodes in this sub-block and map them
    # to our current node-labeling
    block_path = block_prefix + str(block_id)
    assert os.path.exists(block_path), block_path
    nodes = ndist.loadNodes(block_path)

    # TODO we extract the graph locally with ndist
    # the issue with this is that we store the node 2 block list as n5 which could wack the file system ...
    # if we change this storage to hdf5, everything should be fine
    inner_edges, outer_edges, sub_uvs = ndist.extractSubgraphFromNodes(nodes,
                                                                       node_storage_prefix,
                                                                       block_prefix)
    # we might only have a single node, but we still need to find the outer edges
    if len(nodes) <= 1:
        return outer_edges
    assert len(sub_uvs) == len(inner_edges)

    n_local_nodes = int(sub_uvs.max() + 1)
    sub_graph = nifty.graph.undirectedGraph(n_local_nodes)
    sub_graph.insertEdges(sub_uvs)

    sub_costs = costs[inner_edges]
    sub_result = agglomerator(sub_graph, sub_costs)
    sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]

    assert len(sub_edgeresult) == len(inner_edges)
    cut_edge_ids = inner_edges[sub_edgeresult]
    return np.concatenate([cut_edge_ids, outer_edges])


def multicut_step1(block_prefix,
                   node_storage,
                   scale,
                   tmp_folder,
                   agglomerator_key,
                   block_file):

    t0 = time.time()
    agglomerator = AGGLOMERATORS[agglomerator_key]
    costs = z5py.File(os.path.join(tmp_folder, 'problem.n5/s%i' % scale), use_zarr_format=False)['costs'][:]

    block_ids = np.load(block_file)

    node_storage_prefix = os.path.join(node_storage, 'node_')
    cut_edge_ids = np.concatenate([solve_block_subproblem(block_id,
                                                          block_prefix,
                                                          node_storage_prefix,
                                                          costs,
                                                          agglomerator)
                                   for block_id in block_ids])
    cut_edge_ids = np.unique(cut_edge_ids)

    job_id = int(os.path.split(block_file)[1].split('_')[3][:-4])
    np.save(os.path.join(tmp_folder, '1_output_s%i_%i.npy' % (scale, job_id)),
            cut_edge_ids)

    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("block_prefix", type=str)
    parser.add_argument("node_storage", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--agglomerator_key", type=str)
    parser.add_argument("--block_file", type=str)
    args = parser.parse_args()

    multicut_step1(args.block_prefix, args.node_storage,
                   args.scale, args.tmp_folder,
                   args.agglomerator_key,
                   args.block_file)
