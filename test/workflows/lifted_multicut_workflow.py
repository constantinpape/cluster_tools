import os
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py
from cluster_tools import LiftedMulticutSegmentationWorkflow


class TestLiftedMulticutWorkflow(unittest.TestCase):
    input_path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    input_key = 'volumes/affinities'
    ws_key = 'volumes/segmentation/watershed'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    out_path = os.path.join(tmp_folder, 'data.n5')

    target = 'local'
    block_shape = [25, 256, 256]
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        os.makedirs(self.config_folder, exist_ok=True)
        config = LiftedMulticutSegmentationWorkflow.get_config()
        global_config = config['global']
        global_config['shebang'] = self.shebang
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_result(self):
        with z5py.File(self.out_path) as f:
            node_labels = f['node_labels'][:]
            mc = f['volumes/lifted_multicut'][:]
        exp_shape = z5py.File(self.input_path)[self.ws_key].shape

        self.assertEqual(mc.shape, exp_shape)
        unique_nodes = np.unique(node_labels)
        unique_segments = np.unique(mc)
        self.assertTrue(np.allclose(unique_nodes, unique_segments))
        self.assertGreater(len(unique_nodes), 20)

    def test_workflow(self):
        task = LiftedMulticutSegmentationWorkflow
        max_jobs = 8
        t = task(input_path=self.input_path, input_key=self.input_key,
                 ws_path=self.input_path, ws_key=self.ws_key,
                 problem_path=self.out_path, node_labels_key='node_labels',
                 output_path=self.out_path, output_key='volumes/lifted_multicut',
                 lifted_labels_path=self.input_path,
                 lifted_labels_key='volumes/segmentation/groundtruth',
                 lifted_prefix='test', n_scales=1, skip_ws=True,
                 config_dir=self.config_folder, tmp_folder=self.tmp_folder,
                 target=self.target, max_jobs=max_jobs)

        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
