import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py

import nifty.graph.rag as nrag
import nifty.distributed as ndist

try:
    from cluster_tools.graph import GraphWorkflow
    from cluster_tools.cluster_tasks import BaseClusterTask
except ImportError:
    sys.path.append('../..')
    from cluster_tools.graph import GraphWorkflow
    from cluster_tools.cluster_tasks import BaseClusterTask


class TestGraph(unittest.TestCase):
    # input_path = '/g/kreshuk/data/isbi2012_challenge/predictions/watershed.n5'
    input_path = '/home/cpape/Work/data/isbi2012/segmentations/isbi_train_seg.n5'
    input_key = 'volumes/watershed'
    tmp_folder = './tmp'
    output_path = './tmp/graph.n5'
    config_folder = './tmp/configs'
    target = 'local'

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        global_config = BaseClusterTask.default_global_config()
        # global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
        global_config['shebang'] = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
        global_config['block_shape'] = [10, 256, 256]
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_result(self):
        # check shapes
        with z5py.File(self.input_path) as f:
            seg = f[self.input_key][:]
            shape = seg.shape
        with z5py.File(self.output_path) as f:
            shape_ = tuple(f.attrs['shape'])
        self.assertEqual(shape, shape_)

        # check graph
        # compute nifty rag
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)

        # load the graph
        # TODO syntax ?!
        graph = ndist.loadAsUndirectedGraph(os.path.join(self.output_path, 'graph'))

        self.assertEqual(rag.numberOfNodes, graph.numberOfNodes)
        self.assertEqual(rag.numberOfEdges, graph.numberOfEdges)
        self.assertTrue(np.allclose(rag.uvIds(), graph.uvIds()))


    def test_graph(self):
        max_jobs = 8
        ret = luigi.build([GraphWorkflow(input_path=self.input_path,
                                         input_key=self.input_key,
                                         graph_path=self.output_path,
                                         n_scales=1,
                                         config_dir=self.config_folder,
                                         tmp_folder=self.tmp_folder,
                                         target=self.target,
                                         max_jobs=max_jobs)], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
