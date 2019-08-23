import json
import os
import sys
import unittest

import luigi
import z5py
import cluster_tools.evaluation as evaluation
import elf.evaluation as val

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestEvaluation(BaseTest):
    seg_key = 'volumes/segmentation/multicut'
    gt_key = 'volumes/segmentation/groundtruth'

    def metrics(self):
        f = z5py.File(self.input_path)

        ds = f[self.seg_key]
        ds.n_threads = 8
        seg = ds[:]

        ds = f[self.gt_key]
        ds.n_threads = 8
        gt = ds[:]

        vis, vim, ri, _ = val.cremi_score(seg, gt, ignore_gt=[0])
        return vis, vim, ri

    def vi_scores(self):
        f = z5py.File(self.input_path)

        ds = f[self.seg_key]
        ds.n_threads = 8
        seg = ds[:]

        ds = f[self.gt_key]
        ds.n_threads = 8
        gt = ds[:]

        scores = val.object_vi(seg, gt, ignore_gt=[0])
        return scores

    def test_eval(self):
        task = evaluation.EvaluationWorkflow
        res_path = './tmp/res.json'
        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                 target=self.target, max_jobs=self.max_jobs,
                 seg_path=self.input_path, seg_key=self.seg_key,
                 gt_path=self.input_path, gt_key=self.gt_key,
                 output_path=res_path)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        self.assertTrue(os.path.exists(res_path))

        with open(res_path) as f:
            results = json.load(f)
        vis = results['vi-split']
        vim = results['vi-merge']
        ri = results['adapted-rand-error']

        vis_exp, vim_exp, ri_exp = self.metrics()

        self.assertAlmostEqual(vis, vis_exp)
        self.assertAlmostEqual(vim, vim_exp)
        self.assertAlmostEqual(ri, ri_exp)

    def test_object_vis(self):
        task = evaluation.ObjectViWorkflow
        res_path = './tmp/res_objs.json'
        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                 target=self.target, max_jobs=self.max_jobs,
                 seg_path=self.input_path, seg_key=self.seg_key,
                 gt_path=self.input_path, gt_key=self.gt_key,
                 output_path=res_path)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        self.assertTrue(os.path.exists(res_path))

        with open(res_path) as f:
            scores = json.load(f)
        scores_exp = self.vi_scores()

        for gt_id, score in scores.items():
            gt_id = int(gt_id)
            self.assertIn(gt_id, scores_exp)
            score_exp = scores_exp[gt_id]
            self.assertAlmostEqual(score[0], score_exp[0])
            self.assertAlmostEqual(score[1], score_exp[1])


if __name__ == '__main__':
    unittest.main()
