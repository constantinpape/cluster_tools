import sys
import unittest
import numpy as np

import luigi
import nifty.tools as nt
import z5py

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


# TODO also test for writing with label_multisets and offsets
class TestWrite(BaseTest):
    input_key = 'volumes/segmentation/watershed'
    assignment_key = 'assignments'
    output_key = 'data'

    def setUp(self):
        super().setUp()
        f = z5py.File(self.input_path)
        ds = f[self.input_key]
        max_id = ds.attrs['maxId'] + 1
        self.node_labels = np.random.randint(0, 2000, size=max_id)

        f_out = z5py.File(self.output_path)
        f_out.create_dataset(self.assignment_key, data=self.node_labels, compression='gzip',
                             chunks=self.node_labels.shape)

    def check_result(self, node_labels):
        node_labels = self.node_labels

        with z5py.File(self.output_path) as f:
            ds = f[self.output_key]
            ds.n_threads = 8
            res = ds[:]

        with z5py.File(self.input_path) as f:
            ds = f[self.input_path]
            ds.n_threads = 8
            exp = ds[:]
            exp = nt.take(node_labels, exp)

        self.assertEqual(res.shape, exp.shape)
        self.assertTrue(np.array_equal(res, exp))

    def test_write(self):
        from cluster_tools.write import WriteLocal
        task = WriteLocal

        t = task(config_dir=self.config_folder, tmp_folder=self.tmp_folder,
                 max_jobs=self.max_jobs, identifier='test',
                 input_path=self.input_path, input_key=self.input_key,
                 output_path=self.output_path, output_key=self.output_key,
                 assignment_path=self.output_path, assignment_key=self.assignment_key)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)


if __name__ == '__main__':
    unittest.main()
