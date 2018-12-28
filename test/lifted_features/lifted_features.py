import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py

import nifty.tools as nt
import nifty.graph.rag as nrag
import nifty.distributed as ndist

try:
    from cluster_tools.lifted_features import LiftedFeaturesFromNodeLabelsWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.lifted_features import LiftedFeaturesFromNodeLabelsWorkflow


class TestLiftedFeatureWorkflow(unittest.TestCase):
    input_path = '/home/constantin/Work/data/cluster_tools_test_data/test_data.n5'
    ws_key = 'volumes/watershed'
    labels_key = 'volumes/test_labels'
    graph_key = 'graph'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target = 'local'
    block_shape = [10, 256, 256]

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        config = LiftedFeaturesFromNodeLabelsWorkflow.get_config()
        global_config = config['global']
        # global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
        global_config['shebang'] = '#! /home/constantin/Work/software/conda/miniconda3/envs/main/bin/python'
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_result(self):
        out_path = self.tmp_folder + '/lifted_feats.n5'
        with z5py.File(out_path) as f:
            uv_ids = f['lifted_neighborhoods/test'][:]
            costs = f['lifted_feats'][:]
        self.assertEqual(len(uv_ids), len(costs))
        self.assertFalse((uv_ids == 0).all())
        self.assertFalse(np.allclose(costs, 0))

    def test_workflow(self):
        max_jobs = 8
        out_path = self.tmp_folder + '/lifted_feats.n5'
        task = LiftedFeaturesFromNodeLabelsWorkflow
        ret = luigi.build([task(ws_path=self.input_path,
                                ws_key=self.ws_key,
                                labels_path=self.input_path,
                                labels_key=self.labels_key,
                                graph_path=self.input_path,
                                graph_key=self.graph_key,
                                output_path=out_path,
                                output_key='lifted_feats',
                                prefix='test',
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


def make_test_labels():
    input_path = '/home/constantin/Work/data/cluster_tools_test_data/test_data.n5'
    ws_key = 'volumes/watershed'
    labels_key = 'volumes/test_labels'

    with z5py.File(input_path) as f:
        ws = f[ws_key][:]
        ids = np.unique(ws)

        out = np.zeros_like(ws, dtype='uint64')
        for seg_id in ws:
            if np.random.rand() > .8:
                out[ws == seg_id] = np.random.randint(0, 10)

        f.create_dataset(labels_key, data=out, compression='gzip',
                         shape=out.shape, dtype='uint64', chunks=(10, 256, 256))


if __name__ == '__main__':
    # make_test_labels()
    unittest.main()
