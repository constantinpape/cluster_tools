#! /usr/bin/python

import time
import os
import argparse
import numpy as np

import vigra
import z5py
import nifty
import nifty.distributed as ndist


def reduce_scalelevel(scale,
                      tmp_folder,
                      shape,
                      block_shape,
                      new_block_shape,
                      cost_accumulation="sum"):

    # get the number of nodes and uv-ids at this scale level
    # as well as the initial node labeling
    if scale == 0:
        initial_node_labeling = None
    else:
        pass
    n_edges = len(uv_ids)

    # get the costs
    costs = z5py.File(os.path.join(tmp_folder, 'costs.n5'), use_zarr_format=False)['s%i' % scale]

    # load the cut-edge ids from the prev. jobs and make merge edge ids

    # merge node pairs with ufd
    ufd = nifty.ufd.ufd(n_nodes)
    merge_pairs = uv_ids[merge_edge_ids]
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
    edge_mapping = nifty.tools.EdgeMapping(uv_ids, node_labeling, numberOfThreads=8)
    new_uv_ids = edge_mapping.newUvIds()

    new_costs = edge_mapping.mapEdgeValues(costs, cost_accumulation, numberOfThreads=8)
    assert len(new_uv_ids) == len(new_costs)

    print("Reduced graph from", n_nodes, "to", n_new_nodes, "nodes;",
          n_edges, "to", len(new_uv_ids), "edges.")

    # map the new graph (= node labeling and corresponding edges)
    # to the next scale level
    f_nodes = z5py.File('./nodes_to_blocks.n5', use_zarr_format=False)
    f_nodes.create_group('s%i' % (scale + 1,))
    node_out_prefix = './nodes_to_blocks.n5/s%i/node_' % (scale + 1,)
    f_graph = z5py.File('./graph.n5', use_zarr_format=False)
    f_graph.create_group('merged_graphs/s%i' % scale)
    if scale == 0:
        block_in_prefix = './graph.n5/sub_graphs/s%i/block_' % scale
    else:
        block_in_prefix = './graph.n5/merged_graphs/s%i/block_' % scale

    block_out_prefix = './graph.n5/merged_graphs/s%i/block_' % (scale + 1,)

    edge_labeling = edge_mapping.edgeMapping()
    ndist.serializeMergedGraph(block_in_prefix, shape,
                               block_shape, new_block_shape,
                               n_new_nodes,
                               node_labeling, edge_labeling,
                               node_out_prefix, block_out_prefix, 8)

    return n_new_nodes, new_uv_ids, new_costs, new_initial_node_labeling
