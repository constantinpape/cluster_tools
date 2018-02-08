import os
from concurrent import futures
import numpy as np

import vigra
import z5py
import nifty
import cremi_tools.segmentation as cseg
import nifty.distributed as ndist


def solve_subproblems_scalelevel(graph, block_prefix,
                                 costs, node_labeling,
                                 n_blocks, agglomerator):

    # we only return the edges that will be cut
    def solve_subproblem(block_id):
        # load the nodes in this sub-block and map them
        # to our current node-labeling
        block_path = block_prefix + str(block_id)
        assert os.path.exists(block_path), block_path
        nodes = ndist.loadNodes(block_path)

        if node_labeling is not None:
            nodes = nifty.tools.take(node_labeling, nodes)
            nodes = np.unique(nodes)

        # TODO can we do this without loading the whole graph ???
        # TODO make this accecpt and return np array
        inner_edges, outer_edges, sub_graph = graph.extractSubgraphFromNodes(nodes.tolist())
        inner_edges = np.array(inner_edges, dtype='int64')
        outer_edges = np.array(outer_edges, dtype='int64')

        # we might only have a single node, but we still need to find the outer edges
        if len(nodes) <= 1:
            return outer_edges

        sub_costs = costs[inner_edges]
        sub_result = agglomerator(sub_graph, sub_costs)
        sub_uvs = sub_graph.uvIds()
        sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]

        assert len(sub_edgeresult) == len(inner_edges)
        cut_edge_ids = inner_edges[sub_edgeresult]
        return np.concatenate([cut_edge_ids, outer_edges])

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(solve_subproblem, block_id) for block_id in range(n_blocks)]
        cut_edge_ids = np.concatenate([t.result() for t in tasks])

    # cut_edge_ids = np.concatenate([solve_subproblem(block_id) for block_id in range(n_blocks)])

    cut_edge_ids = np.unique(cut_edge_ids)
    merge_edges = np.ones(graph.numberOfEdges, dtype='bool')
    merge_edges[cut_edge_ids] = False
    return np.where(merge_edges)[0]


# actually, we don't even need the graph here, just the uv-ids and number of nodes
# this will allow us to skip graph construction on the cluster nodes :)
def reduce_scalelevel(graph, costs, merge_edge_ids, initial_node_labeling, cost_accumulation="sum"):

    # merge node pairs with ufd
    ufd = nifty.ufd.ufd(graph.numberOfNodes)
    uv_ids = graph.uvIds()
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

    print("Reduced graph from", graph.numberOfNodes, "to", n_new_nodes, "nodes;",
          graph.numberOfEdges, "to", len(new_uv_ids), "edges.")

    # TODO for the clustr version, we only need to serialize
    # the new uv-ids and new number of nodes to serialize the graph
    new_graph = nifty.graph.undirectedGraph(n_new_nodes)
    new_graph.insertEdges(new_uv_ids)
    return new_graph, new_costs, new_initial_node_labeling


def solve_global_problem(graph, costs, initial_node_labeling, agglomerator):
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
    # load / make the inputs
    graph = ndist.loadAsUndirectedGraph(os.path.join(graph_path, graph_key))
    feature_ds = z5py.File(feature_path)['features']
    costs = 1. - feature_ds[:, 0:1].squeeze()
    if weight_edges:
        edge_sizes = feature_ds[:, 9:].squeeze()
    else:
        edge_sizes = None

    # find ignore edges
    ignore_edges = (graph.uvIds() == 0).any(axis=1)

    # set edge sizes of ignore edges to 1 (we don't want them to influence the weighting)
    edge_sizes[ignore_edges] = 1
    costs = cseg.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)

    # set weights of ignore edges to be maximally repulsive
    costs[ignore_edges] = 5 * costs.min()

    initial_node_labeling = None
    shape = z5py.File(graph_path)[graph_key].attrs['shape']
    agglomerator = cseg.Multicut('kernighan-lin')

    for scale in range(n_scales):
        factor = 2**scale
        block_shape = [bs * factor for bs in initial_block_shape]
        blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                        roiEnd=list(shape),
                                        blockShape=block_shape)
        n_blocks = blocking.numberOfBlocks

        block_prefix = os.path.join(graph_path, 'sub_graphs/s%i/block_' % scale)
        print("Solving sub-problems for scale", scale)
        merge_edge_ids = solve_subproblems_scalelevel(graph, block_prefix,
                                                      costs, initial_node_labeling,
                                                      n_blocks, agglomerator)
        print("Merging sub-solutions for scale", scale)
        graph, costs, initial_node_labeling = reduce_scalelevel(graph, costs,
                                                                merge_edge_ids,
                                                                initial_node_labeling)

    initial_node_labeling = solve_global_problem(graph, costs,
                                                 initial_node_labeling, agglomerator)

    out = z5py.File(out_path, use_zarr_format=False)
    if out_key not in out:
        out.create_dataset(out_key, dtype='uint64', shape=tuple(shape),
                           chunks=tuple(initial_block_shape),
                           compression='gzip')
    else:
        # TODO assertions
        pass

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=initial_block_shape)
    n_initial_blocks = blocking.numberOfBlocks
    block_ids = list(range(n_initial_blocks))
    ndist.nodeLabelingToPixels(os.path.join(labels_path, labels_key),
                               os.path.join(out_path, out_key),
                               initial_node_labeling, block_ids,
                               initial_block_shape)
