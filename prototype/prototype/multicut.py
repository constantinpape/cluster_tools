import os
from concurrent import futures
import numpy as np

import z5py
import nifty
import nifty.segmentation as nseg
import nifty.distributed as ndist


def solve_subproblems_scalelevel(graph, block_prefix,
                                 costs, node_labeling,
                                 scale, n_blocks, agglomerator):

    # we only return the edges that will be cut
    def solve_subproblem(block_id):
        # load the nodes in this sub-block and map them
        # to our current node-labeling
        block_path = os.path.join(block_prefix, block_id)
        nodes, _ = ndist.loadNodes(block_path)
        # TODO what is the most efficient here ? nifty.tools.take ?
        if node_labeling is not None:
            nodes = nifty.tools.take(node_labeling, nodes)
            nodes = np.unique(nodes)

        # TODO make this accecpt np array
        inner_edges, outer_edges, sub_graph = graph.extractSubgraphFromNodes(nodes.tolist())

        # we might only have a single node, but we still need to find the outer edges
        if len(nodes) <= 1:
            # TODO cut the outer edges
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

    cut_edge_ids = np.unique(cut_edge_ids)
    merge_edges = np.ones(graph.numberOfEdges, dtype='bool')
    merge_edges[cut_edge_ids] = False
    return np.where(merge_edges)[0]


# TODO implement different merging for costs (e.g. mean)
# TODO compare this implementation with edge contraction graph to my mc_luigi implementation
def reduce_scalelevel(graph, costs, merge_edge_ids, initial_nodes_labeling):
    # build the edge contraction graph
    contraction_graph = nifty.graph.edgeContractionGraph(graph,
                                                         nifty.graph.EdgeContractionGraphCallback())
    #
    # TODO all this should be vectorized / parallelized if possible!
    #
    for merge_edge in merge_edge_ids:
        contraction_graph.contractEdge(merge_edge)

    # get the labeling of initial nodes
    new_initial_nodes_labeling = np.zeros(len(initial_nodes_labeling))
    for node in range(graph.numberOfNodes):
        new_node = contraction_graph.findRepresentativeNode(node)
        initial_node = initial_nodes_labeling[node]
        new_initial_nodes_labeling[initial_node] = new_node

    # get new edge costs
    new_costs = np.zeros(contraction_graph.numberOfEdges, dtype='float32')
    # construct the cut edges from the merge edges
    cut_edges = np.ones(graph.numberOfEdges, dtype='bool')
    cut_edges[merge_edge_ids] = False
    cut_edges = np.where(cut_edges)[0]
    # TODO we want to be able to use different cost mergings here
    for cut_edge in cut_edges:
        new_edge = contraction_graph.findRepresentativeEdge(cut_edge)
        new_costs[new_edge] += costs[cut_edge]

    # this is a bit dumb, for the cluster we should serialize the contraction graph directly
    new_graph = nifty.UndirectedGraph(contraction_graph.numberOfNodes)
    new_graph.insertEdges(contraction_graph.uvIds())
    return new_graph, new_costs, new_initial_nodes_labeling


def solve_global_problem(graph, costs, initial_nodes_labeling, agglomerator):
    node_labeling = agglomerator(graph, costs)

    # TODO vectorize !
    # get the labeling of initial nodes
    new_initial_nodes_labeling = np.zeros(len(initial_nodes_labeling))
    for node in range(graph.numberOfNodes):
        new_node = node_labeling[node]
        initial_node = initial_nodes_labeling[node]
        new_initial_nodes_labeling[initial_node] = new_node

    return new_initial_nodes_labeling


def multicut(graph_path, initial_block_shape, n_scales):

    # TODO load / make the inputs
    graph = ndist.loadAsUndirectedGraph(graph_path)
    costs = z5py.File('./features.n5')['features'][:, 0]
    edge_sizes = z5py.File('./features.n5')['features'][:, 9]
    costs = ndist.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    initial_nodes_labeling = None

    shape = z5py.File(graph_path).attrs['shape']

    agglomerator = nseg.Multicut('kernighan-lin')

    for scale in range(n_scales):
        factor = 2**scale
        block_shape = [bs * factor for bs in initial_block_shape]
        blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                        roiEnd=list(shape),
                                        blockShape=block_shape)
        n_blocks = blocking.numberOfBlocks
        block_prefix = os.path.join(graph_path, 'sub_graphs/s%i/block_' % scale)
        merge_edge_ids = solve_subproblems_scalelevel(graph, block_prefix,
                                                      costs, initial_nodes_labeling,
                                                      scale, n_blocks, agglomerator)
        graph, costs, initial_nodes_labeling = reduce_scalelevel(graph, costs,
                                                                 merge_edge_ids, initial_nodes_labeling)

    initial_nodes_labeling = solve_global_problem(graph, costs,
                                                  initial_nodes_labeling, agglomerator)
