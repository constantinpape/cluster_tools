import os
import json
import unittest
from shutil import rmtree

import luigi
import z5py
import cluster_tools.evaluation as evaluation
import cluster_tools.utils.validation_utils as val


class TestEvaluation(unittest.TestCase):
    path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    seg_key = 'volumes/segmentation/multicut'
    gt_key = 'volumes/segmentation/groundtruth'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target = 'local'
    block_shape = [25, 256, 256]
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'

    def setUp(self):
        os.makedirs(self.config_folder, exist_ok=True)

        configs = evaluation.EvaluationWorkflow.get_config()
        global_config = configs['global']

        global_config['shebang'] = self.shebang
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def _tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def metrics(self):
        f = z5py.File(self.path)

        ds = f[self.seg_key]
        ds.n_threads = 8
        seg = ds[:]

        ds = f[self.gt_key]
        ds.n_threads = 8
        gt = ds[:]

        vis, vim, ri, _ = val.cremi_score(seg, gt, ignore_gt=[0])
        return vis, vim, ri

    def vi_scores(self):
        f = z5py.File(self.path)

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
                 target=self.target, max_jobs=4,
                 seg_path=self.path, seg_key=self.seg_key,
                 gt_path=self.path, gt_key=self.gt_key,
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
                 target=self.target, max_jobs=4,
                 seg_path=self.path, seg_key=self.seg_key,
                 gt_path=self.path, gt_key=self.gt_key,
                 output_path=res_path)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        self.assertTrue(os.path.exists(res_path))

        with open(res_path) as f:
            scores = json.load(f)
        scores_exp = self.vi_scores()

        for gt_id, score in scores.items():
            self.assertIn(gt_id, scores_exp)
            score_exp = scores_exp[gt_id]
            self.assertAlmostEqual(score[0], score_exp[0])
            self.assertAlmostEqual(score[1], score_exp[1])


if __name__ == '__main__':
    unittest.main()
