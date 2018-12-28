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
    from cluster_tools import MulticutSegmentationWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools import MulticutSegmentationWorkflow


class TestMulticutWorkflow(unittest.TestCase):
    input_path = '/home/constantin/Work/data/cluster_tools_test_data/test_data.n5'
    input_key = 'volumes/boundaries_float32'
    ws_key = 'volumes/watershed'
    graph_key = 'graph'
    tmp_folder = './tmp'
    output_path = './tmp/features.n5'
    output_key = 'features'
    config_folder = './tmp/configs'
    target = 'local'
    block_shape = [10, 256, 256]

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        config = MulticutSegmentationWorkflow.get_config()
        global_config = config['global']
        # global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
        global_config['shebang'] = '#! /home/constantin/Work/software/conda/miniconda3/envs/main/bin/python'
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_result(self):
        mc_path = self.tmp_folder + '/mc.n5'
        with z5py.File(mc_path) as f:
            node_labels = f['node_labels'][:]
            mc = f['volumes/multicut'][:]
        self.assertEqual(mc.shape, (30, 512, 512))
        unique_nodes = np.unique(node_labels)
        unique_segments = np.unique(mc)
        self.assertTrue(np.allclose(unique_nodes, unique_segments))
        self.assertGreater(len(unique_nodes), 20)

    def test_workflow(self):
        max_jobs = 8
        mc_path = self.tmp_folder + '/mc.n5'
        ret = luigi.build([MulticutSegmentationWorkflow(input_path=self.input_path,
                                                        input_key=self.input_key,
                                                        ws_path=self.input_path,
                                                        ws_key=self.ws_key,
                                                        problem_path=mc_path,
                                                        node_labels_path=mc_path,
                                                        node_labels_key='node_labels',
                                                        output_path=mc_path,
                                                        output_key='volumes/multicut',
                                                        n_scales=1,
                                                        skip_ws=True,
                                                        config_dir=self.config_folder,
                                                        tmp_folder=self.tmp_folder,
                                                        target=self.target,
                                                        max_jobs=max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
