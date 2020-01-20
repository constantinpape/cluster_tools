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

    def check_features(self, res, expected, feat_name, ids=None):
        expected_feats = expected[feat_name]

        if ids is not None:
            expected_feats = expected_feats[ids]

        self.assertEqual(res.shape, expected_feats.shape)
        self.assertTrue(np.allclose(res, expected_feats))

    def check_result(self, feature_names):
        # load the result
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]
        # compute the vigra result
        with z5py.File(self.input_path) as f:
            inp = f[self.input_key]
            inp.n_threads = self.max_jobs
            inp = normalize(inp[:], 0, 255)

            seg = f[self.seg_key]
            seg.max_jobs = self.max_jobs
            seg = seg[:].astype('uint32')

        expected = vigra.analysis.extractRegionFeatures(inp, seg, features=feature_names,
                                                        ignoreLabel=0)
        for feat_id, feat_name in enumerate(feature_names):
            self.check_features(res[:, feat_id], expected, feat_name)

    def check_subresults(self):
        f_feat = z5py.File(os.path.join(self.tmp_folder, 'region_features_tmp.n5'))
        ds_feat = f_feat['block_feats']
        feature_names = ds_feat.attrs['feature_names']
        n_cols = len(feature_names) + 1

        with z5py.File(self.input_path) as f:
            data = f[self.input_key]
            data.n_threads = self.max_jobs
            data = normalize(data[:], 0, 255)

            segmentation = f[self.seg_key]
            segmentation.max_jobs = self.max_jobs
            segmentation = segmentation[:].astype('uint32')
        blocking = nt.blocking([0, 0, 0], data.shape, self.block_shape)

        n_blocks = blocking.numberOfBlocks
        for block_id in range(n_blocks):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            inp = data[bb]
            seg = segmentation[bb].astype('uint32')

            # load the sub-result
            chunk_id = tuple(beg // bs for beg, bs in zip(block.begin, self.block_shape))
            res = ds_feat.read_chunk(chunk_id)
            if res is None:
                self.assertEqual(seg.sum(), 0)
                continue

            # check that ids are correct
            ids = res[::n_cols].astype('uint32')
            expected_ids = np.unique(seg)
            self.assertEqual(ids.shape, expected_ids.shape)
            self.assertTrue(np.array_equal(ids, expected_ids))

            # check that the features are correct
            expected = vigra.analysis.extractRegionFeatures(inp, seg, features=feature_names,
                                                            ignoreLabel=0)
            for feat_id, feat_name in enumerate(feature_names, 1):
                feat_res = res[feat_id::n_cols]
                self.check_features(feat_res, expected, feat_name, ids)

        return feature_names

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
        feature_names = self.check_subresults()
        self.check_result(feature_names)


if __name__ == '__main__':
    unittest.main()
