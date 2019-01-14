import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py

import nifty.tools as nt
import nifty.graph.rag as nrag
import nifty.distributed as ndist

try:
    from cluster_tools.lifted_features.sparse_lifted_neighborhood import SparseLiftedNeighborhoodLocal
    from cluster_tools.node_labels import NodeLabelWorkflow
    from cluster_tools.graph import GraphWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.lifted_features.sparse_lifted_neighborhood import SparseLiftedNeighborhoodLocal
    from cluster_tools.node_labels import NodeLabelWorkflow
    from cluster_tools.graph import GraphWorkflow


class TestLiftedFeatureWorkflow(unittest.TestCase):
    input_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data_lifted.n5'
    ws_key = 'volumes/watershed'
    labels_key = 'volumes/labels'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    block_shape = [50, 256, 256]

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        global_config = SparseLiftedNeighborhoodLocal.default_global_config()
        global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def _tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def compute_nh(self, graph_depth):
        # load the graph
        graph_path = os.path.join(self.tmp_folder, 'graph.n5', 'graph')
        graph = ndist.Graph(graph_path)
        # load the node labels
        node_label_path = os.path.join(self.tmp_folder, 'node_labels.n5')
        node_label_key = 'node_labels'
        node_labels = z5py.File(node_label_path)[node_label_key][:]
        # run bfs up to depth 4 to get complete lifted nh
        # TODO
        # filter by nodes which have a node labeling
        # TODO
        # return nh

    def _check_result(self, graph_depth):
        # compute nh in memory
        nh = self.compute_nh(graph_depth)
        # load the nh
        out_path = os.path.join(self.tmp_folder, 'lifted_nh.h5')
        out_key = 'lifted_nh'
        nh_out = z5py.File(out_path)[out_key][:]
        # check that results agree
        self.assertEqual(nh_out.shape, nh.shape)

    def test_lifted_nh(self):

        node_label_path = os.path.join(self.tmp_folder, 'node_labels.n5')
        node_label_key = 'node_labels'
        task_labels = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                                        max_jobs=8, target='local',
                                        ws_path=self.input_path, ws_key=self.ws_key,
                                        input_path=self.input_path, input_key=self.labels_key,
                                        output_path=node_label_path, output_key=node_label_key,
                                        max_overlap=True)

        graph_path = os.path.join(self.tmp_folder, 'graph.n5')
        graph_key = 'graph'
        task_graph = GraphWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                                   max_jobs=8, target='local',
                                   input_path=self.input_path, input_key=self.ws_key,
                                   graph_path=graph_path, output_key=graph_key)
        ret = luigi.build([task_labels, task_graph],
                          local_scheduler=True)
        self.assertTrue(ret)

        # TODO try different graph depth and different number of threads !
        graph_depth = 4
        out_path = os.path.join(self.tmp_folder, 'lifted_nh.h5')
        out_key = 'lifted_nh'
        task_nh = SparseLiftedNeighborhoodLocal(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                                                max_jobs=1, dependency=task_graph,
                                                graph_path=graph_path, graph_key=graph_key,
                                                node_label_path=node_label_path, node_label_key=node_label_key,
                                                output_path=out_path, output_key=out_key,
                                                prefix='', nh_graph_depth=graph_depth)
        ret = luigi.build([task_nh], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(graph_depth)


if __name__ == '__main__':
    unittest.main()
