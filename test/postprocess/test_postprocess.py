import os
import unittest
import sys

import luigi
import numpy as np
import nifty
import nifty.graph.rag as nrag
import nifty.distributed as ndist
import vigra
import z5py
from elf.evaluation import rand_index

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestPostprocess(BaseTest):
    def test_size_filter_bg(self):
        from cluster_tools.postprocess import SizeFilterWorkflow
        task = SizeFilterWorkflow
        output_key = "filtered"
        thresh = 250
        t = task(tmp_folder=self.tmp_folder,
                 config_dir=self.config_folder,
                 target=self.target, max_jobs=self.max_jobs,
                 input_path=self.input_path, input_key=self.ws_key,
                 output_path=self.output_path, output_key=output_key,
                 size_threshold=thresh, relabel=False)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)

    def compute_nifty_graph(self):
        with z5py.File(self.input_path) as f:
            ds = f[self.ws_key]
            ds.n_threads = self.max_jobs
            ws = ds[:]
        rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1,
                           numberOfThreads=self.max_jobs)
        graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        graph.insertEdges(rag.uvIds())
        return graph

    def make_assignments(self, g, path, key):
        n_nodes = g.numberOfNodes
        assignments = np.random.randint(1, 100, n_nodes, dtype="uint64")
        with z5py.File(path) as f:
            f.create_dataset(key, data=assignments, chunks=(int(1e5),))
        return assignments

    def test_graph_connected_components(self):
        from cluster_tools.postprocess import ConnectedComponentsWorkflow
        task = ConnectedComponentsWorkflow

        self.compute_graph(ignore_label=False)

        # check the graph again
        g = self.compute_nifty_graph()
        g1 = ndist.Graph(self.output_path, self.graph_key)
        self.assertEqual(g.numberOfNodes, g1.numberOfNodes)
        self.assertEqual(g.numberOfEdges, g1.numberOfEdges)
        self.assertTrue(np.allclose(g.uvIds(), g1.uvIds()))

        assignment_key = "initial_assignments"
        assignments = self.make_assignments(g, self.output_path, assignment_key)

        # compute expected components
        expected = nifty.graph.connectedComponentsFromNodeLabels(g, assignments)
        vigra.analysis.relabelConsecutive(expected, out=expected)

        out_key = "connected_components"
        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                 target=self.target, max_jobs=self.max_jobs,
                 problem_path=self.output_path, graph_key=self.graph_key,
                 assignment_path=self.output_path, assignment_key=assignment_key,
                 output_path=self.output_path, assignment_out_key=out_key)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)

        # load the output components
        with z5py.File(self.output_path) as f:
            results = f[out_key][:]

        # compare
        self.assertEqual(results.shape, expected.shape)
        ri, _ = rand_index(results, expected)
        self.assertAlmostEqual(ri, 0.)

    def toy_problem(self):
        uv_ids = np.array([[0, 1],
                           [1, 2],
                           [2, 3],
                           [3, 4],
                           [4, 5],
                           [5, 6],
                           [6, 7],
                           [7, 8],
                           [8, 9]], dtype="uint64")
        node_labels = np.array([1, 1, 1, 2, 2, 2, 1, 1, 2, 2],
                               dtype="uint64")
        expected_cc = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
                               dtype="uint64")
        return uv_ids, node_labels, expected_cc

    def test_components_dist_toy(self):
        uv_ids, node_labels, expected_cc = self.toy_problem()

        graph_path = os.path.join(self.tmp_folder, "graph.n5")
        with z5py.File(graph_path) as f:
            g = f.create_group("graph")
            g.attrs["numberOfEdges"] = len(uv_ids)
            g.create_dataset("edges", data=uv_ids, chunks=(int(1e5), 2),
                             compression="raw")

        g = ndist.Graph(graph_path, "graph")
        self.assertEqual(g.numberOfNodes, uv_ids.max() + 1)
        self.assertEqual(g.numberOfEdges, len(uv_ids))

        result = ndist.connectedComponentsFromNodes(g, node_labels, True)
        self.assertEqual(len(result), len(expected_cc))

        ri, _ = rand_index(result, expected_cc)
        self.assertAlmostEqual(ri, 0.)

    def test_components_nifty_toy(self):
        uv_ids, node_labels, expected_cc = self.toy_problem()

        g = nifty.graph.undirectedGraph(int(uv_ids.max()) + 1)
        g.insertEdges(uv_ids)
        self.assertEqual(g.numberOfNodes, uv_ids.max() + 1)
        self.assertEqual(g.numberOfEdges, len(uv_ids))

        result = nifty.graph.connectedComponentsFromNodeLabels(g, node_labels)
        self.assertEqual(len(result), len(expected_cc))

        ri, _ = rand_index(result, expected_cc)
        self.assertAlmostEqual(ri, 0.)


if __name__ == "__main__":
    unittest.main()
