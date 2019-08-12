import os
import sys
import json
import unittest

import numpy as np
import vigra
import luigi
import z5py

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


# TODO tests with mask
class TestWatershed(BaseTest):
    input_key = 'volumes/affinities'
    output_key = 'data'

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

    def _run_ws(self, two_pass):
        from cluster_tools.watershed import WatershedWorkflow
        task = WatershedWorkflow(input_path=self.input_path,
                                 input_key=self.input_key,
                                 output_path=self.output_path,
                                 output_key=self.output_key,
                                 config_dir=self.config_folder,
                                 tmp_folder=self.tmp_folder,
                                 target=self.target,
                                 max_jobs=self.max_jobs,
                                 two_pass=two_pass)
        ret = luigi.build([task], local_scheduler=True)
        return ret

    def _test_ws_2d(self, two_pass):
        from cluster_tools.watershed import WatershedLocal
        config = WatershedLocal.default_task_config()
        config['apply_presmooth_2d'] = True
        config['apply_dt_2d'] = True
        config['apply_ws_2d'] = True
        config['threshold'] = 0.25
        config['sigma_weights'] = 0.
        config['halo'] = [0, 32, 32]
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws(two_pass)
        self.assertTrue(ret)
        self._check_result()

    def test_ws_2d(self):
        self._test_ws_2d(False)

    def test_ws_2d_two_pass(self):
        self._test_ws_2d(True)

    def _test_ws_3d(self, two_pass):
        from cluster_tools.watershed import WatershedLocal
        config = WatershedLocal.default_task_config()
        config['apply_presmooth_2d'] = False
        config['apply_dt_2d'] = False
        config['apply_ws_2d'] = False
        config['sigma_seeds'] = (.5, 2., 2.)
        config['sigma_weights'] = (.5, 2., 2.)
        config['halo'] = [2, 32, 32]
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws(two_pass)
        self.assertTrue(ret)
        self._check_result()

    def test_ws_3d(self):
        self._test_ws_3d(False)

    def test_ws_3d_two_pass(self):
        self._test_ws_3d(True)

    def test_ws_pixel_pitch(self):
        from cluster_tools.watershed import WatershedLocal
        config = WatershedLocal.default_task_config()
        config['apply_presmooth_2d'] = False
        config['apply_dt_2d'] = False
        config['apply_ws_2d'] = False
        config['pixel_pitch'] = (10, 1, 1)
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws(False)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
