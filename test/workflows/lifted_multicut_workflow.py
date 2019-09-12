import unittest
import sys
import numpy as np

import luigi
import z5py
from cluster_tools import LiftedMulticutSegmentationWorkflow

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestLiftedMulticutWorkflow(BaseTest):
    input_key = 'volumes/boundaries'
    ws_key = 'volumes/segmentation/watershed'

    def check_result(self):
        with z5py.File(self.output_path) as f:
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
        t = task(input_path=self.input_path, input_key=self.input_key,
                 ws_path=self.input_path, ws_key=self.ws_key,
                 problem_path=self.output_path, node_labels_key='node_labels',
                 output_path=self.output_path, output_key='volumes/lifted_multicut',
                 lifted_labels_path=self.input_path,
                 lifted_labels_key='volumes/segmentation/groundtruth',
                 lifted_prefix='test', n_scales=1, skip_ws=True,
                 config_dir=self.config_folder, tmp_folder=self.tmp_folder,
                 target=self.target, max_jobs=self.max_jobs)

        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        self.check_result()


if __name__ == '__main__':
    unittest.main()
