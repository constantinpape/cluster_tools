import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py
from sklearn.metrics import adjusted_rand_score

try:
    from cluster_tools.mutex_watershed import MwsWorkflow
    from cluster_tools.utils import segmentation_utils as su
except ImportError:
    sys.path.append('../..')
    from cluster_tools.mutex_watershed import MwsWorkflow
    from cluster_tools.utils import segmentation_utils as su


class TestMws(unittest.TestCase):
    # input_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    input_path = '/home/cpape/Work/data/cluster_tools_test_data/test_data.n5'
    input_key = 'volumes/full_affinities'
    mask_key = 'volumes/mask'
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
    strides = [1, 10, 10]

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
        # global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
        global_config['shebang'] = '#! /home/cpape/Work/software/conda/miniconda3/envs/main/bin/python'
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
            exp = su.mutex_watershed(affs, self.isbi_offsets, self.strides,
                                     mask=mask)
            self.assertTrue(np.allclose(exp[np.logical_not(mask)], 0))
            score = adjusted_rand_score(exp.ravel(), res.ravel())
            # score is much better with mask, so most of the differences seem
            # to be due to boundary artifacts
            self.assertLess(1. - score, .01)
        else:
            exp = su.mutex_watershed(affs, self.isbi_offsets, self.strides)
            score = adjusted_rand_score(exp.ravel(), res.ravel())
            self.assertLess(1. - score, .175)

        from cremi_tools.viewer.volumina import view
        view([affs.transpose((1, 2, 3, 0)), res, exp, mask.astype('uint32')],
             ['affs', 'result', 'expected', 'mask'])

    def test_mws(self):
        max_jobs = 8

        config = MwsWorkflow.get_config()['mws_blocks']
        config['strides'] = self.strides
        with open(os.path.join(self.config_folder, 'mws_blocks.config'), 'w') as f:
            json.dump(config, f)

        task = MwsWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                           max_jobs=max_jobs, target=self.target,
                           input_path=self.input_path, input_key=self.input_key,
                           output_path=self.output_path, output_key=self.output_key,
                           offsets=self.isbi_offsets, overlap_threshold=.75)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(with_mask=False)

    def test_mws_with_mask(self):
        max_jobs = 8

        config = MwsWorkflow.get_config()['mws_blocks']
        config['strides'] = self.strides
        with open(os.path.join(self.config_folder, 'mws_blocks.config'), 'w') as f:
            json.dump(config, f)

        task = MwsWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                           max_jobs=max_jobs, target=self.target,
                           input_path=self.input_path, input_key=self.input_key,
                           output_path=self.output_path, output_key=self.output_key,
                           mask_path=self.input_path, mask_key=self.mask_key,
                           offsets=self.isbi_offsets, overlap_threshold=.75)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(with_mask=True)


def add_full_offsets():
    from z5py.converter import convert_from_h5
    in_path = '/g/kreshuk/data/isbi2012_challenge/predictions/isbi2012_train_affinities.h5'
    out_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    print("Copying affs")
    convert_from_h5(in_path, out_path, 'data', 'volumes/full_affinities',
                    n_threads=8)


def make_fake_mask():
    input_path = '/home/cpape/Work/data/cluster_tools_test_data/test_data.n5'
    with z5py.File(input_path) as f:
        shape = f['volumes/raw'].shape
        mask = np.zeros(shape, dtype='uint8')
        central_bb = np.s_[2:-2, 32:-32, 32:-32]
        mask[central_bb] = 1
        f.create_dataset('volumes/mask', data=mask, compression='gzip',
                         chunks=(10, 256, 256))


if __name__ == '__main__':
    # add_full_offsets()
    # make_fake_mask()
    unittest.main()
