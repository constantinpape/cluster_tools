import os
import sys
import json
import unittest

import numpy as np
from skimage.morphology import label
from elf.evaluation import rand_index

import luigi
import z5py
import nifty.tools as nt

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestThresholdedComponents(BaseTest):
    input_key = 'volumes/boundaries'
    output_key = 'data'
    assignment_key = 'assignments'

    def _check_result(self, mode, check_for_equality=True, threshold=.5):
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]
        with z5py.File(self.input_path) as f:
            inp = f[self.input_key][:]

        if mode == 'greater':
            expected = label(inp > threshold)
        elif mode == 'less':
            expected = label(inp < threshold)
        elif mode == 'equal':
            expected = label(inp == threshold)
        self.assertEqual(res.shape, expected.shape)

        if check_for_equality:
            score = rand_index(res, expected)[0]
            self.assertAlmostEqual(score, 0., places=4)

    def _test_mode(self, mode, threshold=.5):
        from cluster_tools.thresholded_components import ThresholdedComponentsWorkflow
        task = ThresholdedComponentsWorkflow(tmp_folder=self.tmp_folder,
                                             config_dir=self.config_folder,
                                             target=self.target, max_jobs=self.max_jobs,
                                             input_path=self.input_path,
                                             input_key=self.input_key,
                                             output_path=self.output_path,
                                             output_key=self.output_key,
                                             assignment_key=self.assignment_key,
                                             threshold=threshold, threshold_mode=mode)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(mode, threshold=threshold)

    def test_greater(self):
        self._test_mode('greater')

    def test_less(self):
        self._test_mode('less')

    def test_equal(self):
        self._test_mode('equal', threshold=0)

    @unittest.skip("debugging test")
    def test_first_stage(self):
        from cluster_tools.thresholded_components.block_components import BlockComponentsLocal
        from cluster_tools.utils.task_utils import DummyTask
        task = BlockComponentsLocal(tmp_folder=self.tmp_folder,
                                    config_dir=self.config_folder,
                                    max_jobs=8,
                                    input_path=self.input_path,
                                    input_key=self.input_key,
                                    output_path=self.output_path,
                                    output_key=self.output_key,
                                    threshold=.5,
                                    dependency=DummyTask())
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result('greater', check_for_equality=False)

    @unittest.skip("debugging test")
    def test_second_stage(self):
        from cluster_tools.thresholded_components.block_components import BlockComponentsLocal
        from cluster_tools.thresholded_components.merge_offsets import MergeOffsetsLocal
        from cluster_tools.utils.task_utils import DummyTask
        task1 = BlockComponentsLocal(tmp_folder=self.tmp_folder,
                                     config_dir=self.config_folder,
                                     max_jobs=8,
                                     input_path=self.input_path,
                                     input_key=self.input_key,
                                     output_path=self.output_path,
                                     output_key=self.output_key,
                                     threshold=.5,
                                     dependency=DummyTask())
        offset_path = './tmp/offsets.json'
        with z5py.File(self.input_path) as f:
            shape = f[self.input_key].shape
        task = MergeOffsetsLocal(tmp_folder=self.tmp_folder,
                                 config_dir=self.config_folder,
                                 max_jobs=8,
                                 shape=shape,
                                 save_path=offset_path,
                                 dependency=task1)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self.assertTrue(os.path.exists(offset_path))

        # checks
        # load offsets from file
        with open(offset_path) as f:
            offsets_dict = json.load(f)
            offsets = offsets_dict['offsets']
            max_offset = int(offsets_dict['n_labels']) - 1

        # load output segmentation
        with z5py.File(self.output_path) as f:
            seg = f[self.output_key][:]

        blocking = nt.blocking([0, 0, 0], list(shape), self.block_shape)
        for block_id in range(blocking.numberOfBlocks):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end)
                       for beg, end in zip(block.begin, block.end))
            segb = seg[bb]
            n_labels = len(np.unique(segb))

            # number of labels from offsets
            if block_id < blocking.numberOfBlocks - 1:
                n_offsets = offsets[block_id + 1] - offsets[block_id]
            else:
                n_offsets = max_offset - offsets[block_id]
            self.assertEqual(n_labels, n_offsets)


if __name__ == '__main__':
    unittest.main()
