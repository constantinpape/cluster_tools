import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py

try:
    from cluster_tools.morphology import MorphologyWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.morphology import MorphologyWorkflow


class TestMorphology(unittest.TestCase):
    input_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    input_key = 'volumes/groundtruth'

    output_path = './tmp/morph.n5'
    output_key = 'data'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target= 'local'

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        self.config = MorphologyWorkflow.get_config()
        global_config = self.config['global']
        global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
        global_config['block_shape'] = [10, 256, 256]
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    # TODO also check bounding box
    def _check_result(self):
        # read the input and compute count und unique ids
        with z5py.File(self.input_path) as f:
            seg = f[self.input_key][:]
        ids, counts = np.unique(seg, return_counts=True)

        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]

        # check correctness for ids and counts / sizes
        self.assertEqual(len(res), len(ids))
        self.assertTrue(np.allclose(ids, res[:, 0]))
        self.assertTrue(np.allclose(counts, res[:, 1]))

        # check correctness for center off mass
        coms = np.zeros((len(ids), 3))
        for label_id in ids:
            coords = np.where(seg == label_id)
            com = [np.mean(coord) for coord in coords]
            coms[label_id] = com
        self.assertTrue(np.allclose(coms, res[:, 2:5]))

    def test_morphology(self):
        max_jobs = 8
        task = MorphologyWorkflow(input_path=self.input_path,
                                  input_key=self.input_key,
                                  output_path=self.output_path,
                                  output_key=self.output_key,
                                  prefix='test',
                                  config_dir=self.config_folder,
                                  tmp_folder=self.tmp_folder,
                                  target=self.target,
                                  max_jobs=max_jobs)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
