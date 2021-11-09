import os
import sys
import json
import unittest
import numpy as np

import luigi
import z5py

import nifty.tools as nt
import nifty.graph.rag as nrag
import nifty.distributed as ndist

try:
    from ..base import BaseTest
except (ValueError, ImportError):
    sys.path.append('..')
    from base import BaseTest


class TestGraph(BaseTest):
    input_key = 'volumes/segmentation/watershed'
    label_multiset_key = 'volumes/segmentation/multicut_label_multiset/s0'
    label_multiset_key_ref = 'volumes/segmentation/multicut'
    output_key = 'graph'

    def check_subresults(self, seg_key):
        f = z5py.File(self.input_path)
        f_out = z5py.File(self.output_path)
        ds_ws = f[seg_key]

        full_graph = ndist.Graph(self.output_path, self.output_key)

        shape = ds_ws.shape
        blocking = nt.blocking([0, 0, 0], list(shape),
                               self.block_shape)

        ds_nodes = f_out["s0/sub_graphs/nodes"]
        ds_edges = f_out["s0/sub_graphs/edges"]
        ds_edge_ids = f_out["s0/sub_graphs/edge_ids"]

        halo = [1, 1, 1]
        for block_id in range(blocking.numberOfBlocks):
            # get the block with the appropriate halo
            # and the corresponding bounding box
            block = blocking.getBlockWithHalo(block_id, halo)
            outer_block, inner_block = block.outerBlock, block.innerBlock
            bb1 = tuple(slice(beg, end) for beg, end in zip(inner_block.begin,
                                                            inner_block.end))
            bb2 = tuple(slice(beg, end) for beg, end in zip(outer_block.begin,
                                                            inner_block.end))
            # load the nodes
            chunk_id = blocking.blockGridPosition(block_id)
            nodes_deser = ds_nodes.read_chunk(chunk_id)

            # load the segmentation and check that the nodes
            # are correct
            seg1 = ds_ws[bb1]
            nodes_ws = np.unique(seg1)
            self.assertTrue(np.array_equal(nodes_ws, nodes_deser))

            # load the edges and construct the graph
            edges = ds_edges.read_chunk(chunk_id)
            if edges is None:
                self.assertEqual(len(nodes_ws), 1)
                continue
            edges = edges.reshape((edges.size // 2, 2))
            graph = ndist.Graph(edges)

            # compute the rag and check that the graph is correct
            seg2 = ds_ws[bb2]

            # check the graph nodes (only if we have edges)
            if graph.numberOfEdges > 0:
                nodes = graph.nodes()
                nodes_ws2 = np.unique(seg2)
                self.assertTrue(np.array_equal(nodes_ws2, nodes))

            rag = nrag.gridRag(seg2, numberOfLabels=int(seg2.max()) + 1)
            # number of nodes in nifty can be larger
            self.assertGreaterEqual(rag.numberOfNodes, graph.numberOfNodes)
            self.assertEqual(rag.numberOfEdges, graph.numberOfEdges)
            uv_ids = graph.uvIds()
            self.assertTrue(np.array_equal(rag.uvIds(), uv_ids))

            if graph.numberOfEdges == 0:
                continue

            # check the edge ids
            edge_ids = ds_edge_ids.read_chunk(chunk_id)
            self.assertEqual(len(edge_ids), graph.numberOfEdges)
            expected_ids = full_graph.findEdges(uv_ids)
            self.assertTrue(np.array_equal(edge_ids, expected_ids))

    def check_result(self, seg_key):
        # check shapes
        with z5py.File(self.input_path) as f:
            seg = f[seg_key]
            seg.n_threads = 8
            seg = seg[:]
            shape = seg.shape
        with z5py.File(self.output_path) as f:
            shape_ = tuple(f[self.graph_key].attrs['shape'])
        self.assertEqual(shape, shape_)

        # check graph
        # compute nifty rag
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)

        # load the graph
        graph = ndist.Graph(self.output_path, self.output_key)

        self.assertEqual(rag.numberOfNodes, graph.numberOfNodes)
        self.assertEqual(rag.numberOfEdges, graph.numberOfEdges)
        self.assertTrue(np.array_equal(rag.uvIds(), graph.uvIds()))

    def test_graph(self):
        from cluster_tools.graph import GraphWorkflow
        task = GraphWorkflow

        task_config = GraphWorkflow.get_config()['initial_sub_graphs']
        task_config['ignore_label'] = False
        with open(os.path.join(self.config_folder, 'initial_sub_graphs.config'),
                  'w') as f:
            json.dump(task_config, f)

        ret = luigi.build([task(input_path=self.input_path,
                                input_key=self.input_key,
                                graph_path=self.output_path,
                                output_key=self.output_key,
                                n_scales=1,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)], local_scheduler=True)
        self.assertTrue(ret)
        self.check_subresults(self.input_key)
        self.check_result(self.input_key)

    def test_graph_label_multiset(self):
        from cluster_tools.graph import GraphWorkflow
        task = GraphWorkflow

        task_config = GraphWorkflow.get_config()['initial_sub_graphs']
        task_config['ignore_label'] = False
        with open(os.path.join(self.config_folder, 'initial_sub_graphs.config'),
                  'w') as f:
            json.dump(task_config, f)

        ret = luigi.build([task(input_path=self.input_path,
                                input_key=self.label_multiset_key,
                                graph_path=self.output_path,
                                output_key=self.output_key,
                                n_scales=1,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)], local_scheduler=True)
        self.assertTrue(ret)
        self.check_subresults(self.label_multiset_key_ref)
        self.check_result(self.label_multiset_key_ref)


if __name__ == '__main__':
    unittest.main()
