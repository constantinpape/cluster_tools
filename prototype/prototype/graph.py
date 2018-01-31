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

    def extract_subgraph(block_id):
        block = blocking.getBlockWithHalo(block_id, halo)
        outer_block, inner_block = block.outerBlock, block.innerBlock
        # we only need the halo into one direction,
        # hence we use the outer-block only for the end coordinate
        begin = inner_block.begin
        end = outer_block.end
        # TODO should we do this hierarchical ?
        # i.e. find the block coordinates along axes and
        # then store the blocks as z/y/x
        # TODO groups for different scale levels
        block_key = 'subgraphs/block_%i' % block_id
        ndist.computeMergeableRegionGraph(path, labels_key,
                                          begin, end,
                                          graph_path, block_key)

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(extract_subgraph, block_id) for block_id in range(blocking.numberOfBlocks)]
        [t.result for t in tasks]


if __name__ == '__main__':
    pass
