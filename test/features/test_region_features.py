import os
import sys
import unittest
import numpy as np

import luigi
import z5py
import vigra
import nifty.tools as nt
from cluster_tools.utils.volume_utils import normalize

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestRegionFeatures(BaseTest):
    input_key = 'volumes/raw/s0'
    seg_key = 'volumes/segmentation/watershed'
    output_key = 'features'

    def _check_features(self, data, labels, res, ids=None, feat_name='mean'):
        expected = vigra.analysis.extractRegionFeatures(data, labels, features=[feat_name])
        expected = expected[feat_name]

        if ids is not None:
            expected = expected[ids]

        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(np.allclose(res, expected))

    def _check_result(self):
        # load the result
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]
        # compute the vigra result
        with z5py.File(self.input_path) as f:
            inp = f[self.input_key]
            inp.n_threads = self.max_jobs
            inp = normalize(inp[:])

            seg = f[self.seg_key]
            seg.max_jobs = self.max_jobs
            seg = seg[:].astype('uint32')
        self._check_features(inp, seg, res)

    def _check_subresults(self):
        with z5py.File(self.input_path) as f:
            data = f[self.input_key]
            data.n_threads = self.max_jobs
            data = normalize(data[:])

            segmentation = f[self.seg_key]
            segmentation.max_jobs = self.max_jobs
            segmentation = segmentation[:].astype('uint32')
        blocking = nt.blocking([0, 0, 0], data.shape, self.block_shape)

        f_feat = z5py.File(os.path.join(self.tmp_folder, 'region_features_tmp.n5'))
        ds_feat = f_feat['block_feats']

        n_blocks = blocking.numberOfBlocks
        for block_id in range(n_blocks):
            # print("Checking block", block_id, "/", n_blocks)
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            inp = data[bb]
            seg = segmentation[bb].astype('uint32')

            # load the sub-result
            chunk_id = tuple(beg // bs for beg, bs in zip(block.begin, self.block_shape))
            res = ds_feat.read_chunk(chunk_id)
            self.assertFalse(res is None)

            # check that ids are correct
            ids = res[::3].astype('uint32')
            expected_ids = np.unique(seg)
            self.assertEqual(ids.shape, expected_ids.shape)
            self.assertTrue(np.allclose(ids, expected_ids))

            # check that mean is correct
            mean = res[2::3]
            self._check_features(inp, seg, mean, ids)

            # check that counts are correct
            counts = res[1::3]
            self._check_features(inp, seg, counts, ids,
                                 feat_name='count')

    def test_region_features(self):
        from cluster_tools.features import RegionFeaturesWorkflow
        ret = luigi.build([RegionFeaturesWorkflow(input_path=self.input_path,
                                                  input_key=self.input_key,
                                                  labels_path=self.input_path,
                                                  labels_key=self.seg_key,
                                                  output_path=self.output_path,
                                                  output_key=self.output_key,
                                                  config_dir=self.config_folder,
                                                  tmp_folder=self.tmp_folder,
                                                  target=self.target,
                                                  max_jobs=self.max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        self._check_subresults()
        self._check_result()


if __name__ == '__main__':
    unittest.main()
