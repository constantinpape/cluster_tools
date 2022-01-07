import os
import sys
import unittest

from skimage.morphology import label
from elf.evaluation import rand_index
from elf.segmentation.utils import normalize_input

import luigi
import z5py

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestConnectedComponents(BaseTest):
    input_key = "volumes/boundaries"
    output_key = "data"
    assignment_key = "assignments"

    def _check_result(self, mode, check_for_equality=True, threshold=.5):
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]
        with z5py.File(self.input_path) as f:
            inp = normalize_input(f[self.input_key][:])

        if mode == "greater":
            expected = label(inp > threshold)
        elif mode == "less":
            expected = label(inp < threshold)
        elif mode == "equal":
            expected = label(inp == threshold)
        self.assertEqual(res.shape, expected.shape)

        # for debugging
        # import napari
        # v = napari.Viewer()
        # v.add_image(inp)
        # v.add_labels(res)
        # v.add_labels(expected)
        # napari.run()

        if check_for_equality:
            score = rand_index(res, expected)[0]
            self.assertAlmostEqual(score, 0.0, places=4)

    def _test_mode(self, mode, threshold=0.5):
        from cluster_tools.connected_components import ConnectedComponentsWorkflow
        task = ConnectedComponentsWorkflow(tmp_folder=self.tmp_folder,
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
        self._test_mode("greater")

    def test_less(self):
        self._test_mode("less")

    def test_equal(self):
        self._test_mode("equal", threshold=0.0)


if __name__ == "__main__":
    unittest.main()
