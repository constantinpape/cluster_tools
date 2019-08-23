import os
import json
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


class TestLinear(BaseTest):
    input_key = 'volumes/raw/s1'
    output_key = 'transformed'

    def _check_result(self, trafo):
        f_in = z5py.File(self.input_path)
        ds_in = f_in[self.input_key]
        ds_in.n_threads = 8
        exp = ds_in[:]

        if len(trafo) == 2:
            a, b = trafo['a'], trafo['b']
            exp = a * exp + b
        else:
            self.assertEqual(len(trafo), len(exp))
            for z in range(len(exp)):
                a, b = trafo[z]['a'], trafo[z]['b']
                exp[z] = a * exp[z] + b

        f_out = z5py.File(self.output_path)
        ds_out = f_out[self.output_key]
        ds_out.n_threads = 8
        res = ds_out[:]

        self.assertEqual(exp.shape, res.shape)
        self.assertTrue(np.allclose(exp, res))

    def test_global_trafo(self):
        from cluster_tools.transformations import LinearTransformationWorkflow
        task = LinearTransformationWorkflow

        trafo_file = os.path.join(self.tmp_folder, 'trafo.json')
        trafo = {'a': 2, 'b': 1}
        with open(trafo_file, 'w') as f:
            json.dump(trafo, f)

        ret = luigi.build([task(input_path=self.input_path,
                                input_key=self.input_key,
                                output_path=self.output_path,
                                output_key=self.output_key,
                                transformation=trafo_file,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(trafo)

    def test_slice_trafo(self):
        from cluster_tools.transformations import LinearTransformationWorkflow
        task = LinearTransformationWorkflow

        n_slices = z5py.File(self.input_path)[self.input_key].shape[0]

        trafo_file = os.path.join(self.tmp_folder, 'trafo.json')
        trafo = {z: {'a': np.random.rand(), 'b': np.random.rand()}
                 for z in range(n_slices)}
        with open(trafo_file, 'w') as f:
            json.dump(trafo, f)

        ret = luigi.build([task(input_path=self.input_path,
                                input_key=self.input_key,
                                output_path=self.output_path,
                                output_key=self.output_key,
                                transformation=trafo_file,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(trafo)


if __name__ == '__main__':
    unittest.main()
