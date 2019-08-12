import unittest
import sys

import luigi
import numpy as np
import nifty
import nifty.graph.rag as nrag
import z5py
from elf.evaluation import rand_index

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestPostprocess(BaseTest):
    ws_key = 'volumes/segmentation/watershed'
    graph_key = 'graph'

    def test_size_filter_bg(self):
        from cluster_tools.postprocess import SizeFilterWorkflow
        task = SizeFilterWorkflow
        output_key = 'filtered'
        thresh = 250
        t = task(tmp_folder=self.tmp_folder,
                 config_dir=self.config_folder,
                 target=self.target, max_jobs=self.max_jobs,
                 input_path=self.input_path, input_key=self.ws_key,
                 output_path=self.output_path, output_key=output_key,
                 size_threshold=thresh, relabel=False)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)

    def compute_grah(self):
        with z5py.File(self.input_path) as f:
            ds = f[self.ws_key]
            ds.n_threads = self.max_jobs
            ws = ds[:]
        rag = nrag.gridRag(ws, numberOfLabels=int(ws.max()) + 1,
                           numberOfThreads=self.max_jobs)
        graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        graph.insertEdges(graph.uvIds())
        return graph

    def make_assignments(self, g, path, key):
        n_nodes = g.numberOfNodes
        assignments = np.random.randint(1, 500, n_nodes)
        with z5py.File(path) as f:
            f.create_dataset(key, data=assignments, chunks=(int(1e5),))
        return assignments

    def test_graph_connected_components(self):
        from cluster_tools.postprocess import ConnectedComponentsWorkflow
        task = ConnectedComponentsWorkflow

        g = self.compute_grah()
        assignment_key = 'initial_assignments'
        assignments = self.make_assignments(g, self.output_path, assignment_key)

        out_key = 'connected_components'
        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                 target=self.target, max_jobs=self.max_jobs,
                 problem_path=self.input_path, graph_key=self.graph_key,
                 assignment_path=self.output_path, assignment_key=assignment_key,
                 output_path=self.output_path, assignment_out_key=out_key)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)

        # load the output components
        with z5py.File(self.output_path) as f:
            results = f[out_key][:]

        # compute expected components
        expected = nifty.graph.connectedComponentsFromNodeLabels(g, assignments)

        # compare
        self.assertEqual(results.shape, expected.shape)
        ri, _ = rand_index(results, expected)
        self.assertAlmostEqual(ri, 1.)


if __name__ == '__main__':
    unittest.main()
