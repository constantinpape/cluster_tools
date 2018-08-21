import unittest
import time
import os
from concurrent import futures
from shutil import rmtree
import numpy as np

import z5py
import nifty
import nifty.graph.rag as nrag
import nifty.distributed as ndist


class TestRegionGraph(unittest.TestCase):
    labels_path = '/home/papec/Work/my_projects/cluster_tools/prototype/test/testdata.n5'
    labels_key = 'watershed'

    def setUp(self):
        assert os.path.exists(self.labels_path)
        assert os.path.exists(os.path.join(self.labels_path, self.labels_key))
        self.shape = z5py.File(self.labels_path)[self.labels_key].shape
        if not os.path.exists('tmpdir'):
            os.mkdir('tmpdir')

    def tearDown(self):
        if os.path.exists('tmpdir'):
            rmtree('tmpdir')

    def load_graph(self, graph_path, graph_key):
        graph_ds = z5py.File(graph_path)[graph_key]
        n_edges = graph_ds.attrs['numberOfEdges']
        graph = ndist.loadAsUndirectedGraph(os.path.join(graph_path, graph_key))
        self.assertEqual(n_edges, graph.numberOfEdges)
        return graph

    def load_nodes(self, graph_path, graph_key):
        graph_ds = z5py.File(graph_path)[graph_key]
        n_nodes = graph_ds.attrs['numberOfNodes']
        nodes = ndist.loadNodes(os.path.join(graph_path, graph_key))
        self.assertEqual(n_nodes, len(nodes))
        return nodes

    def check_nodes(self, labels, nodes):
        actual_nodes = np.unique(labels)
        n_actual = len(actual_nodes)
        self.assertEqual(n_actual, len(nodes))
        self.assertTrue((nodes == actual_nodes).all())

    def check_rag(self, labels, graph):
        uvs = graph.uvIds()
        rag = nrag.gridRag(labels.astype('uint32'), numberOfLabels=int(labels.max()+1))
        uvs_rag = rag.uvIds()
        self.assertEqual(uvs.shape, uvs_rag.shape)
        self.assertTrue((uvs == uvs_rag).all())

    def check_subgraph(self, graph_path, graph_key, labels, bb):
        sub_labels = labels[bb]
        nodes = self.load_nodes(graph_path, graph_key)
        self.check_nodes(sub_labels, nodes)

        # load the graph from file and compare with nifty.rag
        graph = self.load_graph(graph_path, graph_key)
        self.check_rag(sub_labels, graph)

    def extract_initial_subgraphs(self, graph_path, block_shape, labels):
        blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                        roiEnd=list(self.shape),
                                        blockShape=list(block_shape))
        halo = [1, 1, 1]

        f_graph = z5py.File(graph_path, use_zarr_format=False)
        f_graph.create_group('sub_graphs/s0')

        def extract_subgraph(block_id):
            block = blocking.getBlockWithHalo(block_id, halo)
            outer_block, inner_block = block.outerBlock, block.innerBlock
            # we only need the halo into one direction,
            # hence we use the outer-block only for the end coordinate
            begin = inner_block.begin
            end = outer_block.end
            block_key = 'sub_graphs/s0/block_%i' % block_id
            ndist.computeMergeableRegionGraph(self.labels_path, self.labels_key,
                                              begin, end,
                                              graph_path, block_key)

        with futures.ThreadPoolExecutor(8) as tp:
            tasks = [tp.submit(extract_subgraph, block_id) for block_id in range(blocking.numberOfBlocks)]
            [t.result() for t in tasks]

        for block_id in range(blocking.numberOfBlocks):
            sub_key = 'sub_graphs/s0/block_%i' % block_id
            block = blocking.getBlockWithHalo(block_id, halo)
            outer_block, inner_block = block.outerBlock, block.innerBlock
            bb = tuple(slice(b, e) for b, e in zip(inner_block.begin, outer_block.end))
            self.check_subgraph(graph_path, sub_key, labels, bb)
        return blocking.numberOfBlocks

    def merge_subgraphs(self, graph_path, scale, initial_block_shape, labels):
        factor = 2**scale
        previous_factor = 2**(scale - 1)
        block_shape = [factor * bs for bs in initial_block_shape]
        previous_block_shape = [previous_factor * bs for bs in initial_block_shape]

        blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                        roiEnd=list(self.shape),
                                        blockShape=block_shape)
        previous_blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                                 roiEnd=list(self.shape),
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

        halo = [1, 1, 1]
        for block_id in range(blocking.numberOfBlocks):
            print("Checking level 2 block", block_id)
            sub_key = 'sub_graphs/s%i/block_%i' % (scale, block_id)
            block = blocking.getBlockWithHalo(block_id, halo)
            inner_block, outer_block = block.innerBlock, block.outerBlock
            bb = tuple(slice(b, e) for b, e in zip(inner_block.begin, outer_block.end))
            self.check_subgraph(graph_path, sub_key, labels, bb)

        return blocking.numberOfBlocks

    def get_final_graph(self, graph_path, scale, n_blocks):
        ndist.mergeSubgraphs(graph_path,
                             blockPrefix="sub_graphs/s%i/block_" % scale,
                             blockIds=list(range(n_blocks)),
                             outKey='graph',
                             numberOfThreads=8)
        graph = self.load_graph(graph_path, 'graph')
        return graph

    def check_edge_mapping(self, subgraph_path, sub_key, graph):
        subgraph = self.load_graph(subgraph_path, sub_key)
        block_path = os.path.join(subgraph_path, sub_key)
        if not os.path.exists(os.path.join(block_path, 'edgeIds')):
            return
        edge_ids = z5py.File(block_path)['edgeIds'][:]
        self.assertEqual(len(edge_ids), subgraph.numberOfEdges)
        expected_edge_ids = graph.findEdges(subgraph.uvIds())
        self.assertTrue((edge_ids == expected_edge_ids).all())

    def map_edge_ids(self, graph_path, scale, initial_block_shape, graph):
        factor = 2**scale
        block_shape = [factor * bs for bs in initial_block_shape]
        blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                        roiEnd=list(self.shape),
                                        blockShape=block_shape)
        n_blocks = blocking.numberOfBlocks
        ndist.mapEdgeIdsForAllBlocks(graph_path, 'graph',
                                     blockPrefix='sub_graphs/s%i/block_' % scale,
                                     numberOfBlocks=n_blocks,
                                     numberOfThreads=8)
        subgraph_path = os.path.join(graph_path, 'sub_graphs/s%i' % scale)
        for block_id in range(n_blocks):
            self.check_edge_mapping(subgraph_path, 'block_%i' % block_id, graph)

    def test_multiscale_extraction(self):
        graph_path = './tmpdir/graph.n5'
        block_shape = [25, 256, 256]
        labels = z5py.File(self.labels_path)[self.labels_key][:]

        # extract and test initial blocks
        self.extract_initial_subgraphs(graph_path, block_shape, labels)

        # merge and test scale 2 blocks
        n_blocks1 = self.merge_subgraphs(graph_path, 1, block_shape, labels)

        # merge and test final graph
        graph = self.get_final_graph(graph_path, 1, n_blocks1)
        print("Checking complete graph")
        self.check_rag(labels, graph)

        # map and test the edge ids
        self.map_edge_ids(graph_path, 0, block_shape, graph)
        self.map_edge_ids(graph_path, 1, block_shape, graph)

    def test_subgraph(self):
        roi_begin = [0, 0, 0]
        roi_end = [50, 512, 512]
        bb = tuple(slice(rb, re) for rb, re in zip(roi_begin, roi_end))

        # distributed graph extraction
        graph_path = './tmpdir/graph.n5'
        graph_key = 'block0'
        ndist.computeMergeableRegionGraph(self.labels_path, self.labels_key,
                                          roi_begin, roi_end,
                                          graph_path, graph_key)

        # check that nodes were extracted correctly
        labels = z5py.File(self.labels_path)[self.labels_key]
        self.check_subgraph(graph_path, graph_key, labels, bb)

    def _test_performance(self):
        roi_begin = [10, 1000, 1000]
        roi_end = [60, 1512, 1512]
        bb = tuple(slice(rb, re) for rb, re in zip(roi_begin, roi_end))

        times = []

        graph_file = './tmpdir/subgraph.n5'
        for _ in range(10):
            if os.path.exists('./tmpdir'):
                rmtree('./tmpdir')
                os.mkdir('./tmpdir')
            f_graph = z5py.File(graph_file, use_zarr_format=False)
            t0 = time.time()
            ndist.computeMergeableRegionGraph(self.labels_path, self.labels_key,
                                              roi_begin, roi_end,
                                              graph_file, 'block0')
            times.append(time.time() - t0)
        print("Mean extraction time:", np.mean(times), "s")

        # load the graph from file
        nodes, edges = self.load_graph(f_graph, 'block0')

        # check that nodes were extracted correctly
        labels = z5py.File(self.labels_path)[self.labels_key][bb]
        self.check_nodes(labels, nodes)

        self.check_rag(labels, nodes, edges)


if __name__ == '__main__':
    unittest.main()
