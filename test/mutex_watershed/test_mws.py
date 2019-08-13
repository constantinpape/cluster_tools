import os
import sys
import json
import unittest
import numpy as np

import luigi
import z5py
from sklearn.metrics import adjusted_rand_score
from elf.segmentation.mutex_watershed import mutex_watershed

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestMws(BaseTest):
    input_key = 'volumes/affinities'
    mask_key = 'volumes/mask'
    output_key = 'data'
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    strides = [4, 12, 12]

    def _check_result(self, with_mask=False):
        with z5py.File(self.input_path) as f:
            shape = f[self.input_key].shape[1:]
            affs = f[self.input_key][:3]

        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]
        self.assertEqual(res.shape, shape)

        # load affs and compare
        with z5py.File(self.input_path) as f:
            ds = f[self.input_key]
            ds.n_threads = 8
            affs = ds[:]

        if with_mask:
            with z5py.File(self.input_path) as f:
                mask = f[self.mask_key][:]
            self.assertTrue(np.allclose(res[np.logical_not(mask)], 0))
            exp = mutex_watershed(affs, self.offsets, self.strides, mask=mask)
            self.assertTrue(np.allclose(exp[np.logical_not(mask)], 0))
            score = adjusted_rand_score(exp.ravel(), res.ravel())
            # score is much better with mask, so most of the differences seem
            # to be due to boundary artifacts
            self.assertLess(1. - score, .01)
        else:
            exp = mutex_watershed(affs, self.offsets, self.strides)
            score = adjusted_rand_score(exp.ravel(), res.ravel())
            self.assertLess(1. - score, .175)

        # from cremi_tools.viewer.volumina import view
        # view([affs.transpose((1, 2, 3, 0)), res, exp, mask.astype('uint32')],
        #      ['affs', 'result', 'expected', 'mask'])

    def test_mws(self):
        from cluster_tools.mutex_watershed import MwsWorkflow

        config = MwsWorkflow.get_config()['mws_blocks']
        config['strides'] = self.strides
        with open(os.path.join(self.config_folder, 'mws_blocks.config'), 'w') as f:
            json.dump(config, f)

        task = MwsWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                           max_jobs=self.max_jobs, target=self.target,
                           input_path=self.input_path, input_key=self.input_key,
                           output_path=self.output_path, output_key=self.output_key,
                           offsets=self.offsets, overlap_threshold=.75)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(with_mask=False)

    def test_mws_with_mask(self):
        from cluster_tools.mutex_watershed import MwsWorkflow

        config = MwsWorkflow.get_config()['mws_blocks']
        config['strides'] = self.strides
        with open(os.path.join(self.config_folder, 'mws_blocks.config'), 'w') as f:
            json.dump(config, f)

        task = MwsWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                           max_jobs=self.max_jobs, target=self.target,
                           input_path=self.input_path, input_key=self.input_key,
                           output_path=self.output_path, output_key=self.output_key,
                           mask_path=self.input_path, mask_key=self.mask_key,
                           offsets=self.offsets, overlap_threshold=.75)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(with_mask=True)


if __name__ == '__main__':
    unittest.main()
