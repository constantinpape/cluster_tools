import json
import os
import sys
import unittest

import numpy as np
import luigi
import z5py

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestNodeLabels(BaseTest):

    def test_stats(self):
        from cluster_tools.statistics import DataStatisticsWorkflow

        out_path = os.path.join(self.tmp_folder, "stats.json")
        task = DataStatisticsWorkflow(tmp_folder=self.tmp_folder,
                                      config_dir=self.config_folder,
                                      target=self.target, max_jobs=self.max_jobs,
                                      path=self.input_path, key=self.boundary_key,
                                      output_path=out_path)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)

        with open(out_path) as f:
            stats = json.load(f)

        with z5py.File(self.input_path, "r") as f:
            data = f[self.boundary_key][:]

        self.assertAlmostEqual(stats["max"], np.max(data))
        self.assertAlmostEqual(stats["min"], np.min(data))
        self.assertAlmostEqual(stats["mean"], np.mean(data))
        self.assertAlmostEqual(stats["std"], np.std(data))


if __name__ == "__main__":
    unittest.main()
