import os
import sys
import json
import unittest
from shutil import rmtree

import numpy as np
import luigi
import z5py
import nifty.graph.rag as nrag
import nifty.distributed as ndist
import nifty.tools as nt

try:
    from cluster_tools.postprocess import SizeFilterWorkflow
    from cluster_tools.postprocess import FilterLabelsWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.postprocess import SizeFilterWorkflow
    from cluster_tools.postprocess import FilterLabelsWorkflow


class TestPostprocess(unittest.TestCase):
    path = '/home/constantin/Work/data/cluster_tools_test_data/test_data.n5'
    output_path = './tmp/pp.n5'
    input_key = 'volumes/watershed'
    output_key = 'filtered'
    #
    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target= 'local'
    shebang = '#! /home/constantin/Work/software/conda/miniconda3/envs/main/bin/python'

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        global_config = SizeFilterWorkflow.get_config()['global']
        global_config['shebang'] = self.shebang
        global_config['block_shape'] = [10, 256, 256]
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def _tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_size_filter_bg(self):
        task = SizeFilterWorkflow(tmp_folder=self.tmp_folder,
                                  config_dir=self.config_folder,
                                  target=self.target, max_jobs=8,
                                  input_path=self.path, input_key=self.input_key,
                                  output_path=self.output_path, output_key=self.output_key,
                                  size_threshold=100, relabel=False)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)


if __name__ == '__main__':
    unittest.main()
