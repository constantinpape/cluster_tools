import os
import sys
import unittest

import luigi
import numpy as np
import vigra
import z5py

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestRelabel(BaseTest):
    input_key = "volumes/segmentation/watershed"
    output_key = "data"
    assignment_key = "assignments"

    def _check_result(self):
        with z5py.File(self.output_path) as f:
            ds = f[self.output_key]
            ds.n_threads = 4
            res = ds[:]
        with z5py.File(self.input_path) as f:
            ds = f[self.input_key]
            ds.n_threads = 4
            seg = ds[:]
        exp, _, _ = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=True)
        self.assertEqual(exp.shape, res.shape)
        self.assertTrue((np.unique(res) == np.unique(exp)).all())

    def test_relabel(self):
        from cluster_tools.relabel import RelabelWorkflow
        task = RelabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                               max_jobs=self.max_jobs, target=self.target,
                               input_path=self.input_path, input_key=self.input_key,
                               assignment_path=self.output_path,
                               assignment_key=self.assignment_key,
                               output_path=self.output_path, output_key=self.output_key)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == "__main__":
    unittest.main()
