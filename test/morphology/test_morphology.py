import sys
import unittest
import numpy as np

import luigi
import z5py

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestMorphology(BaseTest):
    input_key = 'volumes/segmentation/groundtruth'
    output_key = 'data'

    # TODO also check bounding box
    def _check_result(self):
        # read the input and compute count und unique ids
        with z5py.File(self.input_path) as f:
            seg = f[self.input_key][:]
        ids, counts = np.unique(seg, return_counts=True)

        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]

        # check correctness for ids and counts / sizes
        self.assertEqual(len(res), len(ids))
        self.assertTrue(np.allclose(ids, res[:, 0]))
        self.assertTrue(np.allclose(counts, res[:, 1]))

        # check correctness for center off mass
        coms = np.zeros((len(ids), 3))
        for label_id in ids:
            coords = np.where(seg == label_id)
            com = [np.mean(coord) for coord in coords]
            coms[label_id] = com
        self.assertTrue(np.allclose(coms, res[:, 2:5]))

    def test_morphology(self):
        from cluster_tools.morphology import MorphologyWorkflow
        task = MorphologyWorkflow(input_path=self.input_path,
                                  input_key=self.input_key,
                                  output_path=self.output_path,
                                  output_key=self.output_key,
                                  prefix='test',
                                  config_dir=self.config_folder,
                                  tmp_folder=self.tmp_folder,
                                  target=self.target,
                                  max_jobs=self.max_jobs)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
