import os
from math import ceil
from concurrent import futures

import z5py
import nifty
import nifty.distributed as ndist


def extract_boundary_map_features(graph_path,
                                  data_path, data_key,
                                  labels_path, labels_key,
                                  n_blocks):
    def block_range(block_ids):
        ndist.extractBlockFeaturesFromBoundaryMaps(graph_path, 'sub_graphs/s1/block_',
                                                   data_path, data_key,
                                                   labels_path, labels_key,
                                                   block_ids, './features.z5/blocks')
    n_jobs = 8
    chunk_size = int(ceil(float(n_blocks) / n_jobs))
    block_list = list(range(n_blocks))
    with futures.ThreadPoolExecutor(n_jobs) as tp:
        tasks = [tp.submit(block_range, block_list[i:i + chunk_size])
                 for idx, i in enumerate(range(0, len(block_list), chunk_size))]
        [t.task() for t in tasks]


def merge_features(graph_path, n_blocks):
    features_out = './features.z5'
    ffeats = z5py.File(features_out)
    n_edges = z5py.File(graph_path).attrs['numberOfEdges']
    # chunk size = 128**3
    chunk_size = min(2097152, n_edges)
    ffeats.create_dataset('features', dtype='float32', shape=(n_edges, 10),
                          chunks=(chunk_size, 1), compression='gzip')
    graph_block_prefix = os.path.join(graph_path, 'block_')
    ndist.mergeFeatureBlocks(graph_block_prefix, './features.z5/blocks/block_',
                             features_out, n_blocks, 0, n_edges, 8)


def features(graph_path, data_path, data_key,
             labels_path, labels_key):

    block_shape = [50, 512, 512]
    shape = z5py.File(data_path, data_key).shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_blocks = blocking.numberOfBlocks
    extract_boundary_map_features(graph_path, data_path, data_key,
                                  labels_path, labels_key, n_blocks)

    merge_features(graph_path, n_blocks)
