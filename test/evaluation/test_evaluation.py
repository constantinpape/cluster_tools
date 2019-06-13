import os
import json
import unittest
from shutil import rmtree

import luigi
import z5py
import cluster_tools.evaluation as evaluation
from cluster_tools.utils.validation_utils import SegmentationValidation
from cremi_tools.metrics import adapted_rand, voi


class TestGraph(unittest.TestCase):
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    seg_key = 'segmentation/multicut'
    gt_key = 'segmentation/groundtruth'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target = 'local'
    block_shape = [50, 512, 512]

    def setUp(self):
        os.makedirs(self.config_folder, exist_ok=True)

        configs = evaluation.EvaluationWorkflow.get_config()
        global_config = configs['global']

        # global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
        global_config['shebang'] = '#! /home/pape/Work/software/conda/miniconda3/envs/main/bin/python'
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def _tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def cremi_metrics(self):
        f = z5py.File(self.path)

        ds = f[self.seg_key]
        ds.n_threads = 8
        seg = ds[:]

        ds = f[self.gt_key]
        ds.n_threads = 8
        gt = ds[:]

        rand = adapted_rand(seg, gt)
        vi_split, vi_merge = voi(seg, gt)

        return rand, vi_split, vi_merge

    def test_eval(self):
        task = evaluation.EvaluationWorkflow
        res_path = './tmp/res.json'
        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                 target=self.target, max_jobs=4,
                 seg_path=self.path, seg_key=self.seg_key,
                 gt_path=self.path, gt_key=self.gt_key,
                 out_path=res_path)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        self.assertTrue(os.path.exists(res_path))

        metric = SegmentationValidation(res_path)
        ri = 1. - metric.adapated_rand_score
        vim = metric.vi_merge
        vis = metric.vi_split

        ri_exp, vis_exp, vim_exp = self.cremi_metrics()
        self.assertAlmostEqual(ri, ri_exp)
        self.assertAlmostEqual(vim, vim_exp)
        self.assertAlmostEqual(vis, vis_exp)


if __name__ == '__main__':
    unittest.main()
