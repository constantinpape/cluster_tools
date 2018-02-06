import unittest
import os
from concurrent import futures
from shutil import rmtree
import numpy as np

import z5py
import nifty
import nifty.graph.rag as nrag
import nifty.distributed as ndist


class TestRegionGraph(unittest.TestCase):
    # path = '/home/papec/Work/my_projects/cluster_tools/prototype/test/testdata.n5'
    path = '/home/papec/Work/neurodata_hdd/testdata.n5'
    labels_key = 'watershed'
    xy_key = 'affs_xy'
    full_key = 'full_affs'
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]

    def load_graph(self, graph_path, graph_key):
        graph = ndist.loadAsUndirectedGraph(os.path.join(graph_path, graph_key))
        return graph

    def compute_graph(self):
        halo = [1, 1, 1]

        f_graph = z5py.File(self.graph_path, use_zarr_format=False)
        f_graph.create_group('sub_graphs/s0')

        def extract_subgraph(block_id):
            block = self.blocking.getBlockWithHalo(block_id, halo)
            outer_block, inner_block = block.outerBlock, block.innerBlock
            # we only need the halo into one direction,
            # hence we use the outer-block only for the end coordinate
            begin = inner_block.begin
            end = outer_block.end
            block_key = 'sub_graphs/s0/block_%i' % block_id
            ndist.computeMergeableRegionGraph(self.path, self.labels_key,
                                              begin, end,
                                              self.graph_path, block_key)

        n_threads = 8
        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(extract_subgraph, block_id) for block_id in range(self.blocking.numberOfBlocks)]
            [t.result() for t in tasks]

        n_blocks = self.blocking.numberOfBlocks
        ndist.mergeSubgraphs(self.graph_path,
                             blockPrefix="sub_graphs/s0/block_",
                             blockIds=list(range(n_blocks)),
                             outKey='graph',
                             numberOfThreads=8)

        ndist.mapEdgeIdsForAllBlocks(self.graph_path, 'graph',
                                     blockPrefix='sub_graphs/s0/block_',
                                     numberOfBlocks=n_blocks,
                                     numberOfThreads=8)

    def setUp(self):
        assert os.path.exists(self.path)
        assert os.path.exists(os.path.join(self.path, self.labels_key))
        self.shape = z5py.File(self.path)[self.labels_key].shape
        self.graph_path = './tmpdir/graph.n5'
        self.block_shape = [25, 256, 256]
        self.blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                             roiEnd=list(self.shape),
                                             blockShape=list(self.block_shape))
        if not os.path.exists('tmpdir'):
            os.mkdir('tmpdir')
        self.compute_graph()

    def tearDown(self):
        if os.path.exists('tmpdir'):
            rmtree('tmpdir')

    def check_features(self, feature_path, feature_key, graph_path, graph_key):
        if not os.path.exists(os.path.join(feature_path, feature_key)):
            return
        graph = self.load_graph(graph_path, graph_key)
        features = z5py.File(feature_path)[feature_key][:]
        self.assertEqual(features.shape[0], graph.numberOfEdges)
        self.assertEqual(features.shape[1], 10)
        self.assertTrue(np.isfinite(features).all())

        for ii in range(features.shape[1]):
            mean_feat = np.mean(features[:, ii])
            std_feat = np.std(features[:, ii])
            self.assertNotEqual(mean_feat, 0)
            self.assertNotEqual(std_feat, 0)
            # count feature should never be 0
            if ii == 9:
                self.assertTrue((features[:, ii] != 0).all())

    def test_bmap_features(self):
        print("Test bmap features")
        features_out = './tmpdir/features_bmap.n5'
        ffeats = z5py.File(features_out, use_zarr_format=False)
        ffeats.create_group('blocks')
        features_tmp = os.path.join(features_out, 'blocks')

        def extract_block(block_id):
            ndist.extractBlockFeaturesFromBoundaryMaps(self.graph_path, 'sub_graphs/s0/block_',
                                                       self.path, self.xy_key,
                                                       self.path, self.labels_key,
                                                       [block_id], features_tmp)

        n_blocks = self.blocking.numberOfBlocks
        n_threads = 8
        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(extract_block, block_id) for block_id in range(n_blocks)]
            [t.result() for t in tasks]

        subgraph_path = os.path.join(self.graph_path, 'sub_graphs', 's0')
        for block_id in range(n_blocks):
            # TODO compare to nifty features
            key = 'block_%i' % block_id
            self.check_features(features_tmp, key, subgraph_path, key)

        n_edges = z5py.File(self.graph_path)['graph'].attrs['numberOfEdges']
        chunk_size = min(2097152, n_edges)
        if 'features' not in ffeats:
            ffeats.create_dataset('features', dtype='float32', shape=(n_edges, 10),
                                  chunks=(chunk_size, 1), compression='gzip')

        graph_block_prefix = os.path.join(self.graph_path, 'sub_graphs', 's0', 'block_')
        features_tmp_prefix = os.path.join(features_out, 'blocks/block_')
        edge_offset = 0
        ndist.mergeFeatureBlocks(graph_block_prefix,
                                 features_tmp_prefix,
                                 os.path.join(features_out, 'features'),
                                 n_blocks, edge_offset, n_edges,
                                 numberOfThreads=n_threads)
        self.check_features(features_out, 'features', self.graph_path, 'graph')

    def test_affmap_features(self):
        print("Test affmap features")
        features_out = './tmpdir/features_affmap.n5'
        ffeats = z5py.File(features_out, use_zarr_format=False)
        ffeats.create_group('blocks')
        features_tmp = os.path.join(features_out, 'blocks')

        def extract_block(block_id):
            ndist.extractBlockFeaturesFromAffinityMaps(self.graph_path, 'sub_graphs/s0/block_',
                                                       self.path, self.full_key,
                                                       self.path, self.labels_key,
                                                       [block_id], features_tmp,
                                                       self.offsets)

        n_blocks = self.blocking.numberOfBlocks
        n_threads = 8
        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(extract_block, block_id) for block_id in range(n_blocks)]
            [t.result() for t in tasks]

        subgraph_path = os.path.join(self.graph_path, 'sub_graphs', 's0')
        for block_id in range(n_blocks):
            # TODO compare to nifty features
            key = 'block_%i' % block_id
            self.check_features(features_tmp, key, subgraph_path, key)

        n_edges = z5py.File(self.graph_path)['graph'].attrs['numberOfEdges']
        chunk_size = min(2097152, n_edges)
        if 'features' not in ffeats:
            ffeats.create_dataset('features', dtype='float32', shape=(n_edges, 10),
                                  chunks=(chunk_size, 1), compression='gzip')

        graph_block_prefix = os.path.join(self.graph_path, 'sub_graphs', 's0', 'block_')
        features_tmp_prefix = os.path.join(features_out, 'blocks/block_')
        edge_offset = 0
        ndist.mergeFeatureBlocks(graph_block_prefix,
                                 features_tmp_prefix,
                                 os.path.join(features_out, 'features'),
                                 n_blocks, edge_offset, n_edges,
                                 numberOfThreads=n_threads)
        self.check_features(features_out, 'features', self.graph_path, 'graph')


if __name__ == '__main__':
    unittest.main()
