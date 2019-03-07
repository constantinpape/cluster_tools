import os
import json
import unittest
import numpy as np
from shutil import rmtree

import vigra
import luigi
import z5py

from cluster_tools.relabel import RelabelWorkflow


class TestRelabel(unittest.TestCase):
    input_path = '/home/pape/Work/data/fafb/calyx/central.zarr'
    input_key = 'fragments'
    output_key = 'data'
    assignment_key = 'assignments'
    tmp_folder = './tmp'
    output_path = './tmp/relabeled.n5'
    config_folder = './tmp/configs'
    target = 'local'

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        os.makedirs(self.config_folder, exist_ok=True)
        global_config = RelabelWorkflow.get_config()['global']
        global_config['shebang'] = '#! /home/pape/Work/software/conda/miniconda3/envs/main/bin/python'
        global_config['block_shape'] = [10, 256, 256]
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def _tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_result(self):
        with z5py.File(self.output_path) as f:
            ds = f[self.output_key]
            ds.n_threads = 4
            res = ds[:]
        with z5py.File(self.input_path) as f:
            ds = f[self.input_key]
            ds.n_threads = 4
            seg = ds[:]
        exp, _, _ = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=False)
        self.assertEqual(exp.shape, res.shape)
        self.assertTrue(np.allclose(res, exp))

    def test_relabel(self):
        task = RelabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                               max_jobs=4, target=self.target,
                               input_path=self.input_path, input_key=self.input_key,
                               assignment_path=self.output_path, assignment_key=self.assignment_key,
                               output_path=self.output_path, output_key=self.output_key)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
