import os
import json
import unittest
from shutil import rmtree

import luigi
import cluster_tools.evaluation as evaluation
from cluster_tools.utils.validation_utils import SegmentationValidation


class TestGraph(unittest.TestCase):
    path = '/home/pape/Work/data/cluster_tools_test_data/test_data.n5'
    seg_key = 'volumes/watershed'
    gt_key = 'volumes/groundtruth'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target = 'local'
    block_shape = [10, 256, 256]

    def setUp(self):
        os.makedirs(self.config_folder, exist_ok=True)

        configs = evaluation.EvaluationWorkflow.get_config()
        global_config = configs['global']

        # global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
        global_config['shebang'] = '#! /home/pape/Work/software/conda/miniconda3/envs/main/bin/python'
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_ctable(self):
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
        # TODO compare to independet implementation
        ri = metric.rand_index
        vi = metric.voi
        print(ri)
        print(vi)


if __name__ == '__main__':
    unittest.main()
