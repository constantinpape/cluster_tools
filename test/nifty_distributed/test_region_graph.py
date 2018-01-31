import unittest
import os
from shutil import rmtree
import numpy as np

import z5py
import nifty.graph.rag as nrag
import nifty.distributed as ndist


class TestRegionGraph(unittest.TestCase):
    labels_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sampleA+.n5'
    labels_key = 'watersheds/ws_seeded_z'

    def setUp(self):
        assert os.path.exists(self.labels_path)
        assert os.path.exists(os.path.join(self.labels_path, self.labels_key))
        if not os.path.exists('tmpdir'):
            os.mkdir('tmpdir')

    def tearDown(self):
        if os.path.exists('tmpdir'):
            rmtree('tmpdir')

    def test_subgraph(self):
        roi_begin = [10, 1000, 1000]
        roi_end = [60, 1512, 1512]
        bb = tuple(slice(rb, re) for rb, re in zip(roi_begin, roi_end))

        # distributed graph extraction
        graph_file = './tmpdir/subgraph.n5'
        f_graph = z5py.File(graph_file, use_zarr_format=False)
        ndist.computeMergeableRegionGraph(self.labels_path, self.labels_key,
                                          roi_begin, roi_end,
                                          graph_file, 'block0')
        # load the graph from file
        graph = f_graph['block0']
        n_nodes = graph.attrs['numberOfNodes']
        n_edges = graph.attrs['numberOfEdges']
        nodes = graph['nodes'][:]
        edges = graph['edges'][:]
        self.assertEqual(n_nodes, len(nodes))
        self.assertEqual(n_edges, len(edges))

        # compare to nifty rag
        labels = z5py.File(self.labels_path)[self.labels_key][bb]  # .astype('uint32')
        actual_nodes = np.unique(labels)
        n_actual = len(actual_nodes)
        self.assertEqual(n_actual, n_nodes)
        self.assertTrue((nodes == actual_nodes).all())

        rag = nrag.gridRag(labels.astype('uint32'), numberOfLabels=int(labels.max()+1))

        # we can't naively compare the number of nodes, because nifty rag assumes dense labeling
        uvs = rag.uvIds()
        nodes_rag = np.unique(uvs)
        n_nodes_rag = len(nodes_rag)
        self.assertEqual(n_nodes_rag, n_nodes)
        self.assertTrue((nodes_rag == nodes).all())

        self.assertEqual(edges.shape, uvs.shape)
        self.assertTrue((edges == uvs).all())


if __name__ == '__main__':
    unittest.main()
