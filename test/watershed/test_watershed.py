import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import vigra
import luigi
import z5py

try:
    from cluster_tools.watershed import WatershedWorkflow
    from cluster_tools.watershed.watershed import WatershedLocal
except ImportError:
    sys.path.append('../..')
    from cluster_tools.watershed import WatershedWorkflow
    from cluster_tools.watershed.watershed import WatershedLocal


# TODO tests with mask
class TestWatershed(unittest.TestCase):
    input_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    input_key = 'volumes/affinities_float32'
    output_key = 'data'
    tmp_folder = './tmp'
    output_path = './tmp/ws.n5'
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
        global_config = WatershedLocal.default_global_config()
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

        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:].astype('uint32')

        self.assertEqual(res.shape, shape)
        self.assertFalse(np.allclose(res, 0))
        if with_mask:
            self.assertTrue(0 in res)
        else:
            self.assertFalse(0 in res)
        # make sure that we don't have disconnected segments
        ids0 = np.unique(res)
        res_cc = vigra.analysis.labelVolume(res)
        ids1 = np.unique(res_cc)
        self.assertEqual(len(ids0), len(ids1))

    def _run_ws(self):
        max_jobs = 8
        task = WatershedWorkflow(input_path=self.input_path,
                                 input_key=self.input_key,
                                 output_path=self.output_path,
                                 output_key=self.output_key,
                                 config_dir=self.config_folder,
                                 tmp_folder=self.tmp_folder,
                                 target=self.target,
                                 max_jobs=max_jobs)
        ret = luigi.build([task], local_scheduler=True)
        return ret

    def test_ws_2d(self):
        config = WatershedLocal.default_task_config()
        config['apply_presmooth_2d'] = True
        config['apply_dt_2d'] = True
        config['apply_ws_2d'] = True
        config['threshold'] = 0.25
        config['sigma_weights'] = 0.
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws()
        self.assertTrue(ret)
        self._check_result()

    def test_ws_3d(self):
        config = WatershedLocal.default_task_config()
        config['apply_presmooth_2d'] = False
        config['apply_dt_2d'] = False
        config['apply_ws_2d'] = False
        config['sigma_seeds'] = (.5, 2., 2.)
        config['sigma_weights'] = (.5, 2., 2.)
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws()
        self.assertTrue(ret)
        self._check_result()

    def test_ws_pixel_pitch(self):
        config = WatershedLocal.default_task_config()
        config['apply_presmooth_2d'] = False
        config['apply_dt_2d'] = False
        config['apply_ws_2d'] = False
        config['pixel_pitch'] = (10, 1 , 1)
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws()
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
