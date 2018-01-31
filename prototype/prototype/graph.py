import os
from concurrent import futures
import numpy as np

import z5py
import nifty
import nifty.distributed as ndist


def subgraphs_from_blocks(path, labels_key, blocks, graph_path):
    assert os.path.exists(path), path
    labels = z5py.File(path)[labels_key]
    assert np.dtype(labels.dtype) == np.dtype('uint64')
    shape = labels.shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(blocks))
    halo = [1, 1, 1]

    f_graph = z5py.File(graph_path, use_zarr_format=False)
    f_graph.create_group('sub_graphs/s1')

    def extract_subgraph(block_id):
        print("Extracting subgraph", block_id, "/", blocking.numberOfBlocks)
        block = blocking.getBlockWithHalo(block_id, halo)
        outer_block, inner_block = block.outerBlock, block.innerBlock
        # we only need the halo into one direction,
        # hence we use the outer-block only for the end coordinate
        begin = inner_block.begin
        end = outer_block.end
        # TODO groups for different scale levels
        block_key = 'sub_graphs/s1/block_%i' % block_id
        ndist.computeMergeableRegionGraph(path, labels_key,
                                          begin, end,
                                          graph_path, block_key)

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(extract_subgraph, block_id) for block_id in range(blocking.numberOfBlocks)]
        [t.result for t in tasks]
    return blocking.numberOfBlocks


# only 1 level for now
def compute_region_graph(labels_path, labels_key, blocks, graph_path):
    n_blocks = subgraphs_from_blocks(labels_path, labels_key, blocks, graph_path)
    block_list = list(range(n_blocks))
    ndist.mergeSubgraphs(graph_path, 'sub_graphs/s1', "block_",
                         block_list, "graph")


def load_graph(graph_path, graph_key):
    assert os.path.exists(graph_path)
    f = z5py.File(graph_path)[graph_key]
    nodes = f['nodes'][:]
    edges = f['edges'][:]
    return nodes, edges
