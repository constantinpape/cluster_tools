import os
import sys
import unittest

import luigi
import numpy as np
import z5py
from skimage.measure import regionprops

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestMorphology(BaseTest):
    input_key = "volumes/segmentation/groundtruth"
    output_key = "data"

    def check_result(self):
        # read segmentation and result
        with z5py.File(self.input_path) as f:
            ds = f[self.input_key]
            ds.n_threads = 8
            seg = ds[:]
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]

        # compute regionprops for expected results
        props = regionprops(seg)

        # regionprops does not compute values for the background label (maybe we should not either)
        if res[0, 0] == 0:
            res = res[1:]

        # check ids
        ids = res[:, 0].astype("uint64")
        ids_exp = np.array([rp.label for rp in props], dtype="uint64")
        self.assertEqual(len(ids), len(ids_exp))
        self.assertTrue(np.array_equal(ids, ids_exp))

        # check counts
        counts = res[:, 1].astype("uint64")
        counts_exp = np.array([rp.area for rp in props], dtype="uint64")
        self.assertTrue(np.array_equal(counts, counts_exp))

        # check center of mass
        com = res[:, 2:5]
        com_exp = np.array([rp.centroid for rp in props])
        self.assertTrue(np.allclose(com, com_exp))

        # check bb_min and bb_max
        bb_min = res[:, 5:8].astype("uint64")
        bb_max = res[:, 8:].astype("uint64")

        slices = [rp.slice for rp in props]
        bb_min_exp = np.array([[b.start for b in bb] for bb in slices], dtype="uint64")
        bb_max_exp = np.array([[b.stop - 1 for b in bb] for bb in slices], dtype="uint64")

        self.assertTrue(np.array_equal(bb_min, bb_min_exp))
        self.assertTrue(np.array_equal(bb_max, bb_max_exp))

    def test_morphology(self):
        from cluster_tools.morphology import MorphologyWorkflow
        task = MorphologyWorkflow(input_path=self.input_path,
                                  input_key=self.input_key,
                                  output_path=self.output_path,
                                  output_key=self.output_key,
                                  prefix="test",
                                  config_dir=self.config_folder,
                                  tmp_folder=self.tmp_folder,
                                  target=self.target,
                                  max_jobs=self.max_jobs)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self.check_result()


if __name__ == "__main__":
    unittest.main()
