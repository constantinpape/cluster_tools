import json
import multiprocessing
import os
import unittest
from shutil import rmtree

import luigi
from cluster_tools.cluster_tasks import BaseClusterTask
from cluster_tools.graph import GraphWorkflow

INPUT_PATH = os.environ.get('CLUSTER_TOOLS_TEST_PATH',
                            '/g/kreshuk/data/cremi/example/sampleA.n5')
SHEBANG = os.environ.get('CLUSTER_TOOLS_TEST_SHEBANG',
                         '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python')
MAX_JOBS = os.environ.get('CLUSTER_TOOS_TEST_MAX_JOBS', multiprocessing.cpu_count())
TARGET = os.environ.get('CLUSTER_TOOLS_TEST_TARGET', 'local')


class BaseTest(unittest.TestCase):
    input_path = INPUT_PATH
    shebang = SHEBANG
    max_jobs = MAX_JOBS
    target = TARGET

    tmp_folder = './tmp'
    config_folder = './tmp/config'
    output_path = './tmp/data.n5'
    block_shape = [32, 256, 256]

    graph_key = 'graph'
    ws_key = 'volumes/segmentation/watershed'
    boundary_key = 'volumes/boundaries'
    aff_key = 'volumes/affinities'

    def setUp(self):
        os.makedirs(self.config_folder, exist_ok=True)
        config = BaseClusterTask.default_global_config()
        config.update({'shebang': self.shebang, 'block_shape': self.block_shape})
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def compute_graph(self, ignore_label=True):
        task = GraphWorkflow

        config = task.get_config()['initial_sub_graphs']
        config.update({'ignore_label': ignore_label})
        with open(os.path.join(self.config_folder, 'initial_sub_graphs.config'), 'w') as f:
            json.dump(config, f)

        ret = luigi.build([task(input_path=self.input_path,
                                input_key=self.ws_key,
                                graph_path=self.output_path,
                                output_key=self.graph_key,
                                n_scales=1,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)], local_scheduler=True)
        self.assertTrue(ret)

    def get_target_name(self):
        name_dict = {'local': 'Local', 'slurm': 'Slurm', 'lsf': 'LSF'}
        return name_dict[self.target]
