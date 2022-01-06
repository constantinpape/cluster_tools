import os
import sys
import unittest
import numpy as np

import luigi
import z5py

import nifty.distributed as ndist
import nifty.graph.opt.lifted_multicut as nlmc

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestNHWorkflow(BaseTest):
    labels_key = "volumes/segmentation/multicut"
    node_label_key = "node_labels"

    def setUp(self):
        super().setUp()
        self.compute_graph()

    def compute_nh(self, graph_depth, label_path, label_key):
        # load the graph
        graph = ndist.loadAsUndirectedGraph(self.output_path, self.graph_key)

        # load the node labels
        node_labels = z5py.File(label_path)[label_key][:]
        self.assertEqual(len(node_labels), graph.numberOfNodes)

        # run bfs up to depth 4 to get complete lifted nh
        lifted_graph = nlmc.liftedMulticutObjective(graph)
        lifted_graph.insertLiftedEdgesBfs(graph_depth)
        nh = lifted_graph.liftedUvIds()

        # filter by nodes which have a node labeling
        node_ids = np.arange(len(node_labels))
        nodes_with_label = node_ids[node_labels != 0]
        nh_mask = np.in1d(nh, nodes_with_label).reshape(nh.shape)
        nh_mask = nh_mask.all(axis=1)
        nh = nh[nh_mask]
        # need to lexsort - awkward in numpy ...
        nh = nh[np.lexsort(np.rot90(nh))]
        return nh

    def check_result(self, graph_depth, label_path, label_key):
        # compute nh in memory
        nh = self.compute_nh(graph_depth, label_path, label_key)
        # load the nh
        out_key = "lifted_nh"
        nh_out = z5py.File(self.output_path)[out_key][:]
        # check that results agree
        self.assertEqual(nh_out.shape, nh.shape)
        self.assertTrue(np.allclose(nh_out, nh))

    def test_lifted_nh_with_labels(self):
        import cluster_tools.lifted_features.sparse_lifted_neighborhood as nh_tasks
        from cluster_tools.node_labels import NodeLabelWorkflow

        task = NodeLabelWorkflow
        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                 max_jobs=self.max_jobs, target=self.target,
                 ws_path=self.input_path, ws_key=self.ws_key,
                 input_path=self.input_path, input_key=self.labels_key,
                 output_path=self.output_path, output_key=self.node_label_key,
                 max_overlap=True)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)

        base_name = "SparseLiftedNeighborhood"
        name = base_name + self.get_target_name()
        task = getattr(nh_tasks, name)

        # try different graph depth and different number of threads !
        graph_depth = 3
        out_key = "lifted_nh"
        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder, max_jobs=1,
                 graph_path=self.output_path, graph_key=self.graph_key,
                 node_label_path=self.output_path, node_label_key=self.node_label_key,
                 output_path=self.output_path, output_key=out_key,
                 prefix="", nh_graph_depth=graph_depth)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        self.check_result(graph_depth, self.output_path, self.node_label_key)

    def test_lifted_nh(self):
        import cluster_tools.lifted_features.sparse_lifted_neighborhood as nh_tasks

        graph = ndist.loadAsUndirectedGraph(self.output_path, self.graph_key)
        n_nodes = graph.numberOfNodes
        node_labels = np.ones(n_nodes, dtype="uint64")

        node_label_path = os.path.join(self.tmp_folder, "node_labels.n5")
        node_label_key = "node_labels"
        with z5py.File(node_label_path) as f:
            ds = f.require_dataset(node_label_key, shape=node_labels.shape,
                                   dtype=node_labels.dtype,
                                   chunks=(1000,), compression="gzip")
            ds[:] = node_labels

        base_name = "SparseLiftedNeighborhood"
        name = base_name + self.get_target_name()
        task = getattr(nh_tasks, name)

        # try different graph depth and different number of threads !
        graph_depth = 3
        out_key = "lifted_nh"
        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder, max_jobs=1,
                 graph_path=self.output_path, graph_key=self.graph_key,
                 node_label_path=node_label_path, node_label_key=node_label_key,
                 output_path=self.output_path, output_key=out_key,
                 prefix="", nh_graph_depth=graph_depth)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        self.check_result(graph_depth, node_label_path, node_label_key)


if __name__ == "__main__":
    unittest.main()
