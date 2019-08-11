import os
import json
import unittest
from shutil import rmtree
from cluster_tools.cluster_tasks import BaseClusterTask

INPUT_PATH = os.environ.get('CLUSTER_TOOLS_TEST_PATH',
                            '/g/kreshuk/data/cremi/example/sampleA.n5')
SHEBANG = os.environ.get('CLUSTER_TOOLS_TEST_SHEBANG',
                         '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python')
MAX_JOBS = os.environ.get('CLUSTER_TOOS_TEST_MAX_JOBS', 8)
TARGET = os.environ.get('CLUSTER_TOOLS_TEST_TARGET', 'local')


class BaseTest(unittest.TestCase):
    input_path = INPUT_PATH
    shebang = SHEBANG
    max_jobs = MAX_JOBS
    target = TARGET

    tmp_folder = './tmp'
    config_folder = './tmp/config'
    output_path = './tmp/data.n5'
    block_shape = [25, 256, 256]

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
