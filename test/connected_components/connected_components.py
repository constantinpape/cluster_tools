import os
import sys
import json
import unittest
from shutil import rmtree

import numpy as np
from skimage.morphology import label
from sklearn.metrics import adjusted_rand_score

import luigi
import z5py

try:
    from cluster_tools.connected_components import ConnectedComponentsWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.connected_components import ConnectedComponentsWorkflow


class TestConnectedComponents(unittest.TestCase):
    input_path = '/home/constantin/Work/data/cluster_tools_test_data/test_data.n5'
    input_key = 'volumes/watershed'
    output_path = './tmp/ccs.n5'
    output_key = 'data'
    #
    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target= 'local'
    shebang = '#! /home/constantin/Work/software/conda/miniconda3/envs/main/bin/python'

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        global_config = ConnectedComponentsWorkflow.get_config()['global']
        global_config['shebang'] = self.shebang
        global_config['block_shape'] = [10, 256, 256]
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_result(self):
        # load the max ol result
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]
        with z5py.File(self.input_path) as f:
            seg = f[self.input_key][:]
        expected = label(seg)
        self.assertEqual(res.shape, expected.shape)
        score = adjusted_rand_score(expected.ravel(), res.ravel())
        self.assertEqual(score, 1.)

    def test_ccs(self):
        task = ConnectedComponentsWorkflow(tmp_folder=self.tmp_folder,
                                           config_dir=self.config_folder,
                                           target=self.target, max_jobs=8,
                                           input_path=self.input_path,
                                           input_key=self.input_key,
                                           output_path=self.output_path,
                                           output_key=self.output_key)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
