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
    output_key_ol = 'overlaps'
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

    def _tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _compute_and_check_expected(self, ws, inp, res, exclude=None):
        self.assertFalse((res == 0).all())
        rag = nrag.gridRag(ws, numberOfLabels=int(ws.max() + 1))
        expected = nrag.gridRagAccumulateLabels(rag, inp)

        if exclude is not None:
            res = res[exclude]
            expected = expected[exclude]

        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(np.allclose(res, expected))

    def _ovlps_to_max_ovlps(self, res, max_label_id):
        max_overlaps = np.zeros(max_label_id + 1, dtype='uint64')
        exclude = np.ones(max_label_id + 1, dtype='bool')
        for label_id, ovlps in res.items():
            self.assertTrue(ovlps)
            values, counts = np.array(list(ovlps.keys())), np.array(list(ovlps.values()))
            _, count_counts = np.unique(counts, return_counts=True)

            # if we have agreeing counts, the results may be ambiguous, so we exclude the label from check
            if (count_counts > 1).any():
                exclude[label_id] = False
            max_overlaps[label_id] = values[np.argmax(counts)]
        return max_overlaps, exclude

    def _check_result(self):
        # load the max ol result
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]
            ds_ol = f[self.output_key_ol]
            res_ol, max_label_id = ndist.deserializeOverlapChunk(ds_ol.path, (0,))
        n_labels = len(res)
        self.assertEqual(n_labels, max_label_id + 1)

        # extract the max ol labels
        max_ols, exclude = self._ovlps_to_max_ovlps(res_ol, max_label_id)

        with z5py.File(self.path) as f:
            ws = f[self.ws_key][:]
            inp = f[self.input_key][:]
        self._compute_and_check_expected(ws, inp, res, exclude)
        self._compute_and_check_expected(ws, inp, max_ols, exclude)

    def _check_sub_results(self):
        f = z5py.File(self.path)
        dsw = f[self.ws_key]
        dsi = f[self.input_key]
        block_shape = [10, 256, 256]
        blocking = nt.blocking([0, 0, 0], dsw.shape, block_shape)

        tmp_path = os.path.join(self.output_path, 'label_overlaps_')

        for block_id in range(blocking.numberOfBlocks):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            ws = dsw[bb]
            inp = dsi[bb]

            chunk_id = tuple(beg // bs for beg, bs in zip(block.begin, block_shape))
            res, max_label_id = ndist.deserializeOverlapChunk(tmp_path, chunk_id)

            max_overlaps, exclude = self._ovlps_to_max_ovlps(res, max_label_id)
            self._compute_and_check_expected(ws, inp, max_overlaps, exclude)

    def test_node_labels(self):
        config = NodeLabelWorkflow.get_config()['merge_node_labels']
        config.update({'threads_per_job': 8})
        with open(os.path.join(self.config_folder,
                               'merge_node_labels.config'), 'w') as f:
            json.dump(config, f)

        task1 = NodeLabelWorkflow(tmp_folder=self.tmp_folder,
                                 config_dir=self.config_folder,
                                 target=self.target, max_jobs=8,
                                 ws_path=self.path, ws_key=self.ws_key,
                                 input_path=self.path, input_key=self.input_key,
                                 output_path=self.output_path, output_key=self.output_key)
        task2 = NodeLabelWorkflow(tmp_folder=self.tmp_folder,
                                 config_dir=self.config_folder,
                                 target=self.target, max_jobs=8,
                                 ws_path=self.path, ws_key=self.ws_key,
                                 input_path=self.path, input_key=self.input_key,
                                 output_path=self.output_path, output_key=self.output_key_ol,
                                 max_overlap=False)
        ret = luigi.build([task1, task2], local_scheduler=True)
        self.assertTrue(ret)
        # self._check_sub_results()
        self._check_result()


if __name__ == '__main__':
    unittest.main()
