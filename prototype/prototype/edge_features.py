import os
from concurrent import futures

import z5py
import nifty
import nifty.distributed as ndist


def extract_boundary_map_features(graph_path,
                                  data_path, data_key,
                                  labels_path, labels_key,
                                  n_blocks, features_out):
    features_out_tmp = os.path.join(features_out, 'blocks')

    def extract_block(block_id):
        print("Extracting features for block", block_id)
        ndist.extractBlockFeaturesFromBoundaryMaps(graph_path, 'sub_graphs/s0/block_',
                                                   data_path, data_key,
                                                   labels_path, labels_key,
                                                   [block_id], features_out_tmp)
        print("Done", block_id)
    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(extract_block, block_id) for block_id in range(n_blocks)]
        [t.result() for t in tasks]


def extract_affinity_map_features(graph_path,
                                  data_path, data_key,
                                  labels_path, labels_key,
                                  n_blocks, features_out, offsets):
    features_out_tmp = os.path.join(features_out, 'blocks')

    def extract_block(block_id):
        print("Extracting features for block", block_id)
        ndist.extractBlockFeaturesFromAffinityMaps(graph_path, 'sub_graphs/s0/block_',
                                                   data_path, data_key,
                                                   labels_path, labels_key,
                                                   [block_id], features_out_tmp,
                                                   offsets)
        print("Done", block_id)
    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(extract_block, block_id) for block_id in range(n_blocks)]
        [t.result() for t in tasks]


def merge_features(graph_path, n_blocks, features_out):
    ffeats = z5py.File(features_out)
    n_edges = z5py.File(graph_path)['graph'].attrs['numberOfEdges']
    features_tmp_prefix = os.path.join(features_out, 'blocks/block_')

    # chunk size = 64**3
    chunk_size = min(262144, n_edges)
    if 'features' not in ffeats:
        ffeats.create_dataset('features', dtype='float64', shape=(n_edges, 10),
                              chunks=(chunk_size, 1), compression='gzip')
    graph_block_prefix = os.path.join(graph_path, 'sub_graphs', 's0', 'block_')
    n_threads = 8
    edge_offset = 0
    ndist.mergeFeatureBlocks(graph_block_prefix,
                             features_tmp_prefix,
                             os.path.join(features_out, 'features'),
                             n_blocks, edge_offset, n_edges,
                             numberOfThreads=n_threads)


def features(graph_path, data_path, data_key,
             labels_path, labels_key, features_out, offsets=None):

    ffeats = z5py.File(features_out, use_zarr_format=False)
    if 'blocks' not in ffeats:
        ffeats.create_group('blocks')

    data_shape = z5py.File(data_path)[data_key].shape
    if offsets is None:
        assert len(data_shape) == 3
        shape = z5py.File(data_path)[data_key].shape
    else:
        assert len(data_shape) == 4
        shape = z5py.File(data_path)[data_key].shape[1:]

    block_shape = [25, 256, 256]
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_blocks = blocking.numberOfBlocks

    if offsets is None:
        extract_boundary_map_features(graph_path, data_path, data_key,
                                      labels_path, labels_key, n_blocks, features_out)
    else:
        extract_affinity_map_features(graph_path, data_path, data_key,
                                      labels_path, labels_key, n_blocks, features_out, offsets)

    merge_features(graph_path, n_blocks, features_out)
