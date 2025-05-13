import os
import sys
import json
import unittest
import numpy as np

import luigi
import z5py
import cluster_tools.utils.volume_utils as vu
from sklearn.metrics import adjusted_rand_score
from elf.segmentation.mutex_watershed import mutex_watershed
from elf.segmentation.watershed import apply_size_filter

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestMwsWithMask(BaseTest):
    input_key = "volumes/affinities"
    mask_key = "volumes/mask"
    output_key = "data"
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    strides = [4, 12, 12]

    def _check_result(self, size_filter):
        # load affs and compare
        with z5py.File(self.input_path, "r") as f:
            ds = f[self.input_key]
            ds.n_threads = 4
            affs = vu.normalize(ds[:])
            shape = affs.shape[1:]

        with z5py.File(self.output_path, "r") as f:
            res = f[self.output_key][:]
        self.assertEqual(res.shape, shape)

        with z5py.File(self.input_path, "r") as f:
            mask = f[self.mask_key][:].astype("bool")
        self.assertTrue(np.allclose(res[np.logical_not(mask)], 0))
        exp = mutex_watershed(affs, self.offsets, self.strides, mask=mask)
        if size_filter > 0:
            exp += 1
            exp, _ = apply_size_filter(exp.astype("uint32"), np.max(affs[:3], axis=0), size_filter, [1])
            exp[exp == 1] = 0
        self.assertTrue(np.allclose(exp[np.logical_not(mask)], 0))
        score = adjusted_rand_score(exp.ravel(), res.ravel())
        expected_score = 0.05
        self.assertLess(1. - score, expected_score)

    def test_mws_with_mask(self):
        from cluster_tools.mutex_watershed import MwsWorkflow

        config = MwsWorkflow.get_config()["mws_blocks"]
        config["strides"] = self.strides
        size_filter = config["size_filter"]
        with open(os.path.join(self.config_folder, "mws_blocks.config"), "w") as f:
            json.dump(config, f)

        task = MwsWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                           max_jobs=self.max_jobs, target=self.target,
                           input_path=self.input_path, input_key=self.input_key,
                           output_path=self.output_path, output_key=self.output_key,
                           mask_path=self.input_path, mask_key=self.mask_key,
                           offsets=self.offsets)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(size_filter)


if __name__ == "__main__":
    unittest.main()
