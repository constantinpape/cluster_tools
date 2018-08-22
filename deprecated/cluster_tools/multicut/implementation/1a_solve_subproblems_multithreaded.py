#! /usr/bin/python

import time
import os
import argparse
import numpy as np
from concurrent import futures

import z5py
import cremi_tools.segmentation as cseg
from nifty.graph import undirectedGraph
import nifty.distributed as ndist

# TODO support more agglomerators
AGGLOMERATORS = {"multicut_kl": cseg.Multicut("kernighan-lin")}


def solve_block_subproblem(block_id,
                           graph,
                           block_prefix,
                           costs,
                           agglomerator,
                           shape,
                           block_shape,
                           cut_outer_edges):
    # load the nodes in this sub-block and map them
    # to our current node-labeling
    block_path = block_prefix + str(block_id)
    assert os.path.exists(block_path), block_path
    nodes = ndist.loadNodes(block_path)

    # # the ignore label (== 0) spans a lot of blocks, hence it would slow down our
    # # subgraph extraction, which looks at all the blocks containing the node,
    # # enormously, so we skip it
    # # we make sure that these are cut later
    # if nodes[0] == 0:
    #     nodes = nodes[1:]

    # # if we have no nodes left after, we return none
    # if len(nodes) == 0:
    #     return None

    # # extract the local subgraph
    # inner_edges, outer_edges, sub_uvs = ndist.extractSubgraphFromNodes(nodes,
    #                                                                    block_prefix,
    #                                                                    shape,
    #                                                                    block_shape,
    #                                                                    block_id)
    inner_edges, outer_edges, sub_uvs = graph.extractSubgraphFromNodes(nodes)

    # if we had only a single node (i.e. no edge, return the outer edges)
    if len(nodes) == 1:
        return outer_edges if cut_outer_edges else None

    assert len(sub_uvs) == len(inner_edges)
    assert len(sub_uvs) > 0, str(block_id)

    n_local_nodes = int(sub_uvs.max() + 1)
    sub_graph = undirectedGraph(n_local_nodes)
    sub_graph.insertEdges(sub_uvs)

    sub_costs = costs[inner_edges]
    assert len(sub_costs) == sub_graph.numberOfEdges
    # print(len(sub_costs))

    sub_result = agglomerator(sub_graph, sub_costs)
    sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]

    assert len(sub_edgeresult) == len(inner_edges)
    cut_edge_ids = inner_edges[sub_edgeresult]

    # print("block", block_id, "number cut_edges:", len(cut_edge_ids))
    # print("block", block_id, "number outer_edges:", len(outer_edges))

    if cut_outer_edges:
        cut_edge_ids = np.concatenate([cut_edge_ids, outer_edges])

    return cut_edge_ids


def multicut_step1(graph_path,
                   block_prefix,
                   scale,
                   tmp_folder,
                   agglomerator_key,
                   initial_block_shape,
                   block_file,
                   n_threads,
                   cut_outer_edges=True):

    t0 = time.time()
    agglomerator = AGGLOMERATORS[agglomerator_key]
    costs = z5py.File(os.path.join(tmp_folder, 'merged_graph.n5/s%i' % scale),
                      use_zarr_format=False)['costs'][:]

    block_ids = np.load(block_file)

    shape = z5py.File(graph_path).attrs['shape']
    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    # TODO we should have symlinks instead of the if else
    if scale == 0:
        graph_path_ = os.path.join(graph_path, 'graph')
    else:
        graph_path_ = os.path.join(tmp_folder, 'merged_graph.n5', 's%i' % scale)
    graph = ndist.Graph(graph_path_)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(solve_block_subproblem,
                           block_id,
                           graph,
                           block_prefix,
                           costs,
                           agglomerator,
                           shape,
                           block_shape,
                           cut_outer_edges)
                 for block_id in block_ids]
        results = [t.result() for t in tasks]

    results = [res for res in results if res is not None]
    if len(results) > 0:
        cut_edge_ids = np.concatenate(results)
        cut_edge_ids = np.unique(cut_edge_ids).astype('uint64')
    else:
        cut_edge_ids = np.zeros(0, dtype='uint64')

    job_id = int(os.path.split(block_file)[1].split('_')[3][:-4])
    np.save(os.path.join(tmp_folder, '1_output_s%i_%i.npy' % (scale, job_id)),
            cut_edge_ids)

    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("block_prefix", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--agglomerator_key", type=str)
    parser.add_argument("--initial_block_shape", type=int, nargs=3)
    parser.add_argument("--block_file", type=str)
    parser.add_argument("--n_threads", type=int)
    args = parser.parse_args()

    multicut_step1(args.graph_path,
                   args.block_prefix,
                   args.scale, args.tmp_folder,
                   args.agglomerator_key,
                   args.initial_block_shape,
                   args.block_file,
                   args.n_threads)
