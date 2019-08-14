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


class TestWatershed(BaseTest):
    input_key = 'volumes/affinities'
    mask_key = 'volumes/mask'
    output_key = 'watershed'

    def _check_result(self, with_mask):
        with z5py.File(self.input_path) as f:
            shape = f[self.input_key].shape[1:]

        with z5py.File(self.output_path) as f:
            res = f[self.output_key]
            res.n_threads = self.max_jobs
            res = res[:].astype('uint32')

        self.assertEqual(res.shape, shape)
        self.assertFalse(np.allclose(res, 0))
        if with_mask:
            self.assertNotIn(0, res)
        else:
            self.assertIn(0, res)
        # make sure that we don't have disconnected segments
        ids0 = np.unique(res)
        res_cc = vigra.analysis.labelVolume(res)
        ids1 = np.unique(res_cc)
        self.assertEqual(len(ids0), len(ids1))

    def _run_ws(self, with_mask, two_pass):
        from cluster_tools.watershed import WatershedWorkflow
        if with_mask:
            mask_path = self.input_path
            mask_key = self.mask_key
        else:
            mask_path = mask_key = ''
        task = WatershedWorkflow(input_path=self.input_path,
                                 input_key=self.input_key,
                                 output_path=self.output_path,
                                 output_key=self.output_key,
                                 mask_path=mask_path,
                                 mask_key=mask_key,
                                 config_dir=self.config_folder,
                                 tmp_folder=self.tmp_folder,
                                 target=self.target,
                                 max_jobs=self.max_jobs,
                                 two_pass=two_pass)
        ret = luigi.build([task], local_scheduler=True)
        return ret

    def _test_ws_2d(self, with_mask, two_pass):
        from cluster_tools.watershed import WatershedWorkflow
        config = WatershedWorkflow.get_config()['watershed']
        config['apply_presmooth_2d'] = True
        config['apply_dt_2d'] = True
        config['apply_ws_2d'] = True
        config['threshold'] = 0.25
        config['sigma_weights'] = 0.
        config['halo'] = [0, 32, 32]
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws(with_mask, two_pass)
        self.assertTrue(ret)
        self._check_result(with_mask)

    def test_ws_2d(self):
        self._test_ws_2d(with_mask=True, two_pass=False)

    def test_ws_2d_two_pass(self):
        self._test_ws_2d(with_mask=True, two_pass=True)

    def _test_ws_3d(self, with_mask, two_pass):
        from cluster_tools.watershed import WatershedWorkflow
        config = WatershedWorkflow.get_config()['watershed']
        config['apply_presmooth_2d'] = False
        config['apply_dt_2d'] = False
        config['apply_ws_2d'] = False
        config['sigma_seeds'] = (.5, 2., 2.)
        config['sigma_weights'] = (.5, 2., 2.)
        config['halo'] = [2, 32, 32]
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws(with_mask, two_pass)
        self.assertTrue(ret)
        self._check_result(with_mask)

    def test_ws_3d(self):
        self._test_ws_3d(with_mask=True, two_pass=False)

    def test_ws_3d_two_pass(self):
        self._test_ws_3d(with_mask=True, two_pass=False)

    def test_ws_pixel_pitch(self):
        from cluster_tools.watershed import WatershedWorkflow
        config = WatershedWorkflow.get_config()['watershed']
        config['apply_presmooth_2d'] = False
        config['apply_dt_2d'] = False
        config['apply_ws_2d'] = False
        config['pixel_pitch'] = (10, 1, 1)
        with open(os.path.join(self.config_folder, 'watershed.config'), 'w') as f:
            json.dump(config, f)
        ret = self._run_ws(with_mask=True, two_pass=False)
        self.assertTrue(ret)
        self._check_result(with_mask=True)

    def test_no_mask_3d(self):
        self._test_ws_3d(with_mask=False, two_pass=False)

    def test_no_mask_2d(self):
        self._test_ws_2d(with_mask=False, two_pass=False)


if __name__ == '__main__':
    unittest.main()
