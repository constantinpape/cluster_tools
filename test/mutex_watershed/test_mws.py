import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py

from affogato.segmentation import compute_mws_segmentation
from cremi_tools.viewer.volumina import view

try:
    from cluster_tools.mutex_watershed import MwsWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.mutex_watershed import MwsWorkflow


# TODO tests with mask
class TestMws(unittest.TestCase):
    input_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    input_key = 'volumes/full_affinities'
    tmp_folder = './tmp'
    output_path = './tmp/mws.n5'
    output_key = 'data'
    config_folder = './tmp/configs'
    target= 'local'
    isbi_offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                    [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                    [0, -9, 0], [0, 0, -9],
                    [0, -9, -9], [0, 9, -9], [0, -9, -4],
                    [0, -4, -9], [0, 4, -9], [0, 9, -4],
                    [0, -27, 0], [0, 0, -27]]

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        config = MwsWorkflow.get_config()
        global_config = config['global']
        global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
        global_config['block_shape'] = [10, 256, 256]
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_result(self, with_mask=False):
        with z5py.File(self.input_path) as f:
            shape = f[self.input_key].shape[1:]
            affs = f[self.input_key][:3]

        # TODO load affs and compute expected mws
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:].astype('uint32')

        view([affs.transpose((1, 2, 3, 0)), res])
        self.assertEqual(res.shape, shape)
        # TODO compare to expected mws

    def test_mws(self):
        max_jobs = 8
        config = MwsWorkflow.get_config()['mws_blocks']
        config['strides'] = [1, 10, 10]
        with open(os.path.join(self.config_folder, 'mws_blocks.config'), 'w') as f:
            json.dump(config, f)

        task = MwsWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                           max_jobs=max_jobs, target=self.target,
                           input_path=self.input_path, input_key=self.input_key,
                           output_path=self.output_path, output_key=self.output_key,
                           offsets=self.isbi_offsets)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


def add_full_offsets():
    from z5py.converter import convert_from_h5
    in_path = '/g/kreshuk/data/isbi2012_challenge/predictions/isbi2012_train_affinities.h5'
    out_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    print("Copying affs")
    convert_from_h5(in_path, out_path, 'data', 'volumes/full_affinities',
                    n_threads=8)


if __name__ == '__main__':
    # add_full_offsets()
    unittest.main()
