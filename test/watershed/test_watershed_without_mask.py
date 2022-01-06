import os
import sys
import json
import unittest

import numpy as np
import vigra
import luigi
import z5py

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestWatershedWithoutMask(BaseTest):
    input_key = "volumes/affinities"
    output_key = "watershed"

    def _check_result(self):
        with z5py.File(self.input_path) as f:
            shape = f[self.input_key].shape[1:]

        with z5py.File(self.output_path) as f:
            res = f[self.output_key]
            res.n_threads = self.max_jobs
            res = res[:].astype("uint32")

        self.assertEqual(res.shape, shape)
        self.assertNotIn(0, res)

        # make sure that we don"t have disconnected segments
        ids0 = np.unique(res)
        res_cc = vigra.analysis.labelVolume(res)
        ids1 = np.unique(res_cc)
        self.assertEqual(len(ids0), len(ids1))

    def _run_ws(self, two_pass):
        from cluster_tools.watershed import WatershedWorkflow
        mask_path = mask_key = ""
        task = WatershedWorkflow(input_path=self.input_path,
                                 input_key=self.input_key,
                                 output_path=self.output_path,
                                 output_key=self.output_key,
                                 mask_path=mask_path,
                                 mask_key=mask_key,
                                 config_dir=self.config_folder,
                                 tmp_folder=self.tmp_folder,
                                 target=self.target,
                                 max_jobs=self.max_jobs,
                                 two_pass=two_pass)
        ret = luigi.build([task], local_scheduler=True)
        return ret

    def _test_ws_2d(self,  two_pass):
        from cluster_tools.watershed import WatershedWorkflow
        config = WatershedWorkflow.get_config()["watershed"]
        config["apply_presmooth_2d"] = True
        config["apply_dt_2d"] = True
        config["apply_ws_2d"] = True
        config["threshold"] = 0.25
        config["sigma_weights"] = 0.
        config["halo"] = [0, 32, 32]
        with open(os.path.join(self.config_folder, "watershed.config"), "w") as f:
            json.dump(config, f)
        ret = self._run_ws(two_pass)
        self.assertTrue(ret)
        self._check_result()

    def test_ws_2d(self):
        self._test_ws_2d(two_pass=False)

    def _test_ws_3d(self, two_pass):
        from cluster_tools.watershed import WatershedWorkflow
        config = WatershedWorkflow.get_config()["watershed"]
        config["apply_presmooth_2d"] = False
        config["apply_dt_2d"] = False
        config["apply_ws_2d"] = False
        config["sigma_seeds"] = (.5, 2., 2.)
        config["sigma_weights"] = (.5, 2., 2.)
        config["halo"] = [2, 32, 32]
        with open(os.path.join(self.config_folder, "watershed.config"), "w") as f:
            json.dump(config, f)
        ret = self._run_ws(two_pass)
        self.assertTrue(ret)
        self._check_result()

    def test_ws_3d(self):
        self._test_ws_3d(two_pass=False)


if __name__ == "__main__":
    unittest.main()
