import os
import sys
import json
import unittest
from shutil import rmtree

import numpy as np
import luigi
import z5py
import nifty.graph.rag as nrag

try:
    from cluster_tools.node_labels import NodeLabelWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.node_labels import NodeLabelWorkflow


class TestNodeLabels(unittest.TestCase):
    path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    output_path = './tmp/node_labels.n5'
    ws_key = 'volumes/watershed'
    input_key = 'volumes/groundtruth'
    output_key = 'labels'
    #
    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target= 'local'
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        global_config = NodeLabelWorkflow.get_config()['global']
        global_config['shebang'] = self.shebang
        global_config['block_shape'] = [10, 256, 256]
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_result(self):
        pass

    def test_node_labels(self):
        config = NodeLabelWorkflow.get_config()['merge_node_labels']
        config.update({'threads_per_job': 8})
        with open(os.path.join(self.config_folder,
                               'merge_node_labels.config'), 'w') as f:
            json.dump(config, f)

        task = NodeLabelWorkflow(tmp_folder=self.tmp_folder,
                                 config_dir=self.config_folder,
                                 target=self.target, max_jobs=8,
                                 ws_path=self.path, ws_key=self.ws_key,
                                 input_path=self.path, input_key=self.input_key,
                                 output_path=self.output_path, output_key=self.output_key)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
