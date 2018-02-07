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
    f_graph.create_group('sub_graphs/s0')

    def extract_subgraph(block_id):
        print("Extracting subgraph", block_id, "/", blocking.numberOfBlocks)
        block = blocking.getBlockWithHalo(block_id, halo)
        outer_block, inner_block = block.outerBlock, block.innerBlock
        # we only need the halo into one direction,
        # hence we use the outer-block only for the end coordinate
        begin = inner_block.begin
        end = outer_block.end
        # TODO groups for different scale levels
        block_key = 'sub_graphs/s0/block_%i' % block_id
        ndist.computeMergeableRegionGraph(path, labels_key,
                                          begin, end,
                                          graph_path, block_key)

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(extract_subgraph, block_id) for block_id in range(blocking.numberOfBlocks)]
        [t.result for t in tasks]
    return blocking.numberOfBlocks, shape


def merge_subgraphs(graph_path, scale, initial_block_shape, shape):
    factor = 2**scale
    previous_factor = 2**(scale - 1)
    block_shape = [factor * bs for bs in initial_block_shape]
    previous_block_shape = [previous_factor * bs for bs in initial_block_shape]

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    previous_blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                             roiEnd=list(shape),
                                             blockShape=previous_block_shape)

    def merge_block(block_id):
        block = blocking.getBlock(block_id)
        output_key = 'sub_graphs/s%i/block_%i' % (scale, block_id)
        block_list = previous_blocking.getBlockIdsInBoundingBox(roiBegin=block.begin,
                                                                roiEnd=block.end,
                                                                blockHalo=[0, 0, 0])
        ndist.mergeSubgraphs(graph_path,
                             blockPrefix="sub_graphs/s%i/block_" % (scale - 1),
                             blockIds=block_list.tolist(),
                             outKey=output_key)

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(merge_block, block_id) for block_id in range(blocking.numberOfBlocks)]
        [t.result() for t in tasks]
    return blocking.numberOfBlocks


def compute_region_graph(labels_path, labels_key, blocks, graph_path):
    n_blocks0, shape = subgraphs_from_blocks(labels_path, labels_key, blocks, graph_path)
    n_blocks1 = merge_subgraphs('./graph.n5', 1, blocks, shape)

    block_list0 = list(range(n_blocks0))
    block_list1 = list(range(n_blocks1))

    ndist.mergeSubgraphs(graph_path, 'sub_graphs/s1/block_',
                         block_list1, "graph")

    ndist.mapEdgeIds(graph_path, 'graph', 'sub_graphs/s0/block_', block_list0)
    ndist.mapEdgeIds(graph_path, 'graph', 'sub_graphs/s1/block_', block_list1)

    z5py.File('./graph.n5')['graph'].attrs['shape'] = shape
