import os
import numpy as np

import z5py
import nifty
import nifty.distributed as ndist
import nifty.graph.rag as nrag

from cluster_tools.utils.segmentation_utils import multicut_kernighan_lin
from cremi_tools.viewer.volumina import view


# debug starting from the costs that were calculated
# -> PASSES
def debug_costs():
    example_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'

    costs = z5py.File(example_path)['costs'][:]
    edges = z5py.File(example_path)['graph/edges'][:]
    n_nodes = int(edges.max()) + 1
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(edges)

    assert graph.numberOfEdges == len(costs)
    node_labels = multicut_kernighan_lin(graph, costs)

    ws = z5py.File(example_path)['volumes/watersheds'][:]
    rag = nrag.gridRag(ws, numberOfLabels=n_nodes)
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labels)

    view([ws, seg])


# debug starting from the multicut node labels
# -> FAILS
# -> something during multicut goes wrong
def debug_node_labels():
    example_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'

    node_labels = z5py.File(example_path)['node_labels'][:]

    # print(node_labels[:10])
    # quit()

    node_labels = node_labels[:, 1]
    node_labels = np.concatenate((np.zeros(1, dtype='uint64'),
                                  node_labels))
    n_nodes = len(node_labels)

    ws = z5py.File(example_path)['volumes/watersheds'][:]
    rag = nrag.gridRag(ws, numberOfLabels=n_nodes)
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labels)

    view([ws, seg])


def debug_subresult(block_id=1):
    example_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'
    block_prefix = os.path.join(example_path, 's0', 'sub_graphs', 'block_')

    graph = ndist.Graph(os.path.join(example_path, 'graph'))
    block_path = block_prefix + str(block_id)
    nodes = ndist.loadNodes(block_path)
    inner_edges, outer_edges, sub_uvs = graph.extractSubgraphFromNodes(nodes)

    block_res_path = './tmp/subproblem_results/s0_block%i.npy' % block_id
    res = np.load(block_res_path)

    merge_edges = np.ones(graph.numberOfEdges, dtype='bool')
    merge_edges[res] = False
    merge_edges[outer_edges] = False

    uv_ids = graph.uvIds()
    n_nodes = int(uv_ids.max()) + 1
    ufd = nifty.ufd.ufd(n_nodes)
    ufd.merge(uv_ids[merge_edges])
    node_labels = ufd.elementLabeling()

    ws = z5py.File(example_path)['volumes/watersheds'][:]
    rag = nrag.gridRag(ws, numberOfLabels=n_nodes)
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labels)
    view([ws, seg])


def debug_reduce_problem():
    example_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'

    node_labels = z5py.File(example_path)['s1']['node_labeling'][:]
    node_labels = np.concatenate((np.zeros(1, dtype='uint64'),
                                  node_labels))
    n_nodes = len(node_labels)

    ws = z5py.File(example_path)['volumes/watersheds'][:]
    rag = nrag.gridRag(ws, numberOfLabels=n_nodes)
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labels)

    view([ws, seg])


if __name__ == '__main__':
    # debug_node_labels()
    # debug_subresult()
    debug_reduce_problem()
