import sys
import unittest

import luigi
import z5py
from elf.util import downscale_shape

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestDownscaling(BaseTest):
    input_key = 'volumes/raw/s0'
    output_key_prefix = 'data'

    def check_result(self, shape, scales):
        f = z5py.File(self.output_path)
        g = f[self.output_key_prefix]
        self.assertTrue(g.attrs['multiScale'])

        expected_shape = shape
        effective_scale = [1, 1, 1]
        for level, scale in enumerate(scales):
            key = 's%i' % level
            ds = g[key]
            shape_level = ds.shape
            expected_shape = downscale_shape(expected_shape, scale)
            self.assertEqual(shape_level, expected_shape)

            effective_scale = [sc * eff for sc, eff in zip(scale, effective_scale)]
            if level > 0:
                self.assertEqual(effective_scale[::-1], ds.attrs["downsamplingFactors"])

    def test_downscaling(self):
        from cluster_tools.downscaling import DownscalingWorkflow
        task = DownscalingWorkflow

        scales = [[1, 2, 2], [1, 2, 2], [2, 2, 2]]
        halos = [[1, 4, 4], [1, 4, 4], [2, 4, 4]]

        t = task(input_path=self.input_path, input_key=self.input_key,
                 output_path=self.output_path, output_key_prefix=self.output_key_prefix,
                 scale_factors=scales, halos=halos,
                 config_dir=self.config_folder, tmp_folder=self.tmp_folder,
                 target=self.target, max_jobs=self.max_jobs)

        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        shape = z5py.File(self.input_path)[self.input_key].shape
        scales = [[1, 1, 1]] + scales
        self.check_result(shape, scales)


if __name__ == '__main__':
    unittest.main()
