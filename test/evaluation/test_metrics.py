import unittest
import numpy as np

import z5py
from cremi_tools.metrics import adapted_rand, voi


class TestMetrics(unittest.TestCase):
    path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    seg_key = 'volumes/segmentation/multicut'
    gt_key = 'volumes/segmentation/groundtruth'

    # this takes quite a while, so we only do it on a sub-cutout
    # bb = np.s_[:30, :256, :256]
    # bb = np.s_[:]
    bb = np.s_[:100, :1024, :1024]

    def test_vi(self):
        from cluster_tools.utils.validation_utils import variation_of_information
        f = z5py.File(self.path)

        ds_gt = f[self.gt_key]
        ds_gt.n_threads = 4
        gt = ds_gt[self.bb]

        ds_seg = f[self.seg_key]
        ds_seg.n_threads = 4
        seg = ds_seg[self.bb]

        vi_s, vi_m = variation_of_information(seg, gt, ignore_gt=[0])
        vi_s_exp, vi_m_exp = voi(seg, gt)

        self.assertAlmostEqual(vi_s, vi_s_exp)
        self.assertAlmostEqual(vi_m, vi_m_exp)

    def test_ri(self):
        from cluster_tools.utils.validation_utils import rand_index
        f = z5py.File(self.path)

        ds_gt = f[self.gt_key]
        ds_gt.n_threads = 4
        gt = ds_gt[self.bb]

        ds_seg = f[self.seg_key]
        ds_seg.n_threads = 4
        seg = ds_seg[self.bb]

        ari, ri = rand_index(seg, gt, ignore_gt=[0])
        ari_exp = adapted_rand(seg, gt)

        self.assertAlmostEqual(ari, ari_exp)

    def test_cremi_score(self):
        from cluster_tools.utils.validation_utils import cremi_score
        f = z5py.File(self.path)

        ds_gt = f[self.gt_key]
        ds_gt.n_threads = 4
        gt = ds_gt[self.bb]

        ds_seg = f[self.seg_key]
        ds_seg.n_threads = 4
        seg = ds_seg[self.bb]

        vis, vim, ari, cs = cremi_score(seg, gt, ignore_gt=[0])

        ari_exp = adapted_rand(seg, gt)
        vis_exp, vim_exp = voi(seg, gt)

        cs_exp = np.sqrt(ari_exp * (vis_exp + vim_exp))

        self.assertAlmostEqual(ari, ari_exp)
        self.assertAlmostEqual(vis, vis_exp)
        self.assertAlmostEqual(vim, vim_exp)
        self.assertAlmostEqual(cs, cs_exp)


if __name__ == '__main__':
    unittest.main()
