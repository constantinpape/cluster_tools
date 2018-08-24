import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py

from failing_task import FailingTaskLocal


class TestRetry(unittest.TestCase):
    output_path = './tmp/out.n5'
    output_key =  'data'
    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    shape = (100, 1024, 1024)

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        global_config = FailingTaskLocal.default_global_config()
        global_config['shebang'] = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
        global_config['block_shape'] = [10, 256, 256]
        global_config['max_num_retries'] = 1
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_ws_2d(self):
        max_jobs = 8
        ret = luigi.build([FailingTaskLocal(output_path=self.output_path, output_key=self.output_key,
                                            shape=self.shape,
                                            config_dir=self.config_folder,
                                            tmp_folder=self.tmp_folder,
                                            max_jobs=max_jobs)], local_scheduler=True)
        self.assertTrue(ret)
        with z5py.File(self.output_path) as f:
            data = f[self.output_key][:]
        self.assertTrue(np.allclose(data, 1))


if __name__ == '__main__':
    unittest.main()
