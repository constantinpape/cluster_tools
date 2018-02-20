import os
from concurrent import futures
import numpy as np

import vigra
import z5py
import nifty
import cremi_tools.segmentation as cseg
import nifty.distributed as ndist


def solve_subproblems_scalelevel(block_prefix,
                                 node_storage_prefix,
                                 costs,
                                 n_blocks,
                                 agglomerator):

    # we only return the edges that will be cut
    def solve_subproblem(block_id):
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

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(solve_subproblem, block_id) for block_id in range(n_blocks)]
        cut_edge_ids = np.concatenate([t.result() for t in tasks])

    # cut_edge_ids = np.concatenate([solve_subproblem(block_id) for block_id in range(n_blocks)])

    n_edges = len(costs)
    cut_edge_ids = np.unique(cut_edge_ids)
    merge_edges = np.ones(n_edges, dtype='bool')
    merge_edges[cut_edge_ids] = False
    return np.where(merge_edges)[0]


def reduce_scalelevel(scale, n_nodes, uv_ids, costs,
                      merge_edge_ids,
                      initial_node_labeling,
                      shape, block_shape, new_block_shape,
                      cost_accumulation="sum"):

    n_edges = len(uv_ids)
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


def solve_global_problem(n_nodes, uv_ids, costs, initial_node_labeling, agglomerator):
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    node_labeling = agglomerator(graph, costs)

    # get the labeling of initial nodes
    if initial_node_labeling is None:
        new_initial_node_labeling = node_labeling
    else:
        # should this ever become a bottleneck, we can parallelize this in nifty
        # but for now this would really be premature optimization
        new_initial_node_labeling = node_labeling[initial_node_labeling]

    return new_initial_node_labeling


def multicut(labels_path, labels_key,
             graph_path, graph_key,
             feature_path,
             out_path, out_key,
             initial_block_shape, n_scales,
             weight_edges=True):

    assert os.path.exists(feature_path), feature_path

    # graph = ndist.loadAsUndirectedGraph(os.path.join(graph_path, graph_key))
    # load number of nodes and uv-ids
    f_graph = z5py.File(graph_path)[graph_key]
    shape = f_graph.attrs['shape']
    n_nodes = f_graph.attrs['numberOfNodes']
    uv_ids = f_graph['edges'][:]

    # get the multicut edge costs from mean affinities
    feature_ds = z5py.File(feature_path)['features']
    costs = 1. - feature_ds[:, 0:1].squeeze()
    if weight_edges:
        edge_sizes = feature_ds[:, 9:].squeeze()
    else:
        edge_sizes = None

    # find ignore edges
    ignore_edges = (uv_ids == 0).any(axis=1)

    # set edge sizes of ignore edges to 1 (we don't want them to influence the weighting)
    edge_sizes[ignore_edges] = 1
    costs = cseg.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)

    # set weights of ignore edges to be maximally repulsive
    costs[ignore_edges] = 5 * costs.min()

    # get the number of initial blocks
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=initial_block_shape)
    n_initial_blocks = blocking.numberOfBlocks

    # get node to block assignment for scale level 0 and the oversegmentaton nodes
    f_nodes = z5py.File('./nodes_to_blocks.n5', use_zarr_format=False)
    if 's0' not in f_nodes:
        f_nodes.create_group('s0')
    print("Here")
    ndist.nodesToBlocks(os.path.join(graph_path, 'sub_graphs/s0/block_'),
                        os.path.join('./nodes_to_blocks.n5/s0', 'node_'),
                        n_initial_blocks, n_nodes, 8)
    print("There")

    initial_node_labeling = None
    agglomerator = cseg.Multicut('kernighan-lin')

    for scale in range(n_scales):
        factor = 2**scale
        block_shape = [bs * factor for bs in initial_block_shape]
        blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                        roiEnd=list(shape),
                                        blockShape=block_shape)
        n_blocks = blocking.numberOfBlocks

        if scale == 0:
            block_prefix = os.path.join(graph_path, 'sub_graphs/s%i/block_' % scale)
        else:
            block_prefix = os.path.join(graph_path, 'merged_graphs/s%i/block_' % scale)
        print("Solving sub-problems for scale", scale)
        merge_edge_ids = solve_subproblems_scalelevel(block_prefix,
                                                      os.path.join('./nodes_to_blocks.n5', 's%i' % scale, 'node_'),
                                                      costs,
                                                      n_blocks,
                                                      agglomerator)
        print("Merging sub-solutions for scale", scale)
        next_factor = 2**(scale + 1)
        next_block_shape = [bs * next_factor for bs in initial_block_shape]
        n_nodes, uv_ids, costs, initial_node_labeling = reduce_scalelevel(scale, n_nodes, uv_ids, costs,
                                                                          merge_edge_ids,
                                                                          initial_node_labeling,
                                                                          shape, block_shape, next_block_shape)

    initial_node_labeling = solve_global_problem(n_nodes, uv_ids, costs,
                                                 initial_node_labeling, agglomerator)

    out = z5py.File(out_path, use_zarr_format=False)
    if out_key not in out:
        out.create_dataset(out_key, dtype='uint64', shape=tuple(shape),
                           chunks=tuple(initial_block_shape),
                           compression='gzip')
    else:
        # TODO assertions
        pass

    block_ids = list(range(n_initial_blocks))
    ndist.nodeLabelingToPixels(os.path.join(labels_path, labels_key),
                               os.path.join(out_path, out_key),
                               initial_node_labeling, block_ids,
                               initial_block_shape)
