import os
import sys
import json
import unittest
from shutil import rmtree

import numpy as np
import luigi
import z5py

try:
    from cluster_tools.skeletons import SkeletonWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.skeletons import SkeletonWorkflow


class TestSkeletons(unittest.TestCase):
    # path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    path = '/home/constantin/Work/data/cluster_tools_test_data/test_data.n5'

    input_prefix = 'volumes/segmentation'
    output_path = './tmp/skeletons.n5'
    output_prefix = 'skeletons'
    #
    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target= 'local'

    # shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
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
        global_config = SkeletonWorkflow.get_config()['global']
        global_config['shebang'] = self.shebang
        global_config['block_shape'] = [10, 256, 256]
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _run_skel_wf(self, format_, max_jobs):
        task = SkeletonWorkflow(tmp_folder=self.tmp_folder,
                                config_dir=self.config_folder,
                                target=self.target, max_jobs=max_jobs,
                                input_path=self.path, input_prefix=self.input_prefix,
                                output_path=self.output_path, output_prefix=self.output_prefix,
                                work_scale=0, skeleton_format=format_)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        f = z5py.File(self.output_path)
        self.assertTrue(self.output_prefix in f)
        out_key = os.path.join(self.output_prefix, 's0')
        self.assertTrue(out_key in f)

    def test_skeletons_n5(self):
        config = SkeletonWorkflow.get_config()['skeletonize']
        config.update({'chunk_len': 50})
        with open(os.path.join(self.config_folder, 'skeletonize.config'), 'w') as f:
            json.dump(config, f)

        self._run_skel_wf(format_='n5', max_jobs=8)
        # TODO check output for correctness

    def test_skeletons_swc(self):
        config = SkeletonWorkflow.get_config()['skeletonize']
        config.update({'chunk_len': 50})
        with open(os.path.join(self.config_folder, 'skeletonize.config'), 'w') as f:
            json.dump(config, f)

        self._run_skel_wf(format_='swc', max_jobs=8)
        # TODO check output for correctness

    def test_skeletons_volume(self):
        config = SkeletonWorkflow.get_config()['skeletonize']
        config.update({'threads_per_job': 8})
        with open(os.path.join(self.config_folder, 'skeletonize.config'), 'w') as f:
            json.dump(config, f)

        self._run_skel_wf(format_='volume', max_jobs=1)
        # TODO check output for correctness


if __name__ == '__main__':
    unittest.main()
