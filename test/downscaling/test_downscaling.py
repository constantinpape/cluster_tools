import os
import sys
import unittest

import numpy as np
import luigi
import h5py
import z5py
from elf.util import downscale_shape

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestDownscaling(BaseTest):
    input_key = "volumes/raw/s0"
    output_key_prefix = "data"

    def check_result_paintera(self, shape, scales):
        f = z5py.File(self.output_path)
        g = f[self.output_key_prefix]
        self.assertTrue(g.attrs["multiScale"])

        expected_shape = shape
        effective_scale = [1, 1, 1]
        for level, scale in enumerate(scales):
            key = "s%i" % level
            ds = g[key]
            shape_level = ds.shape
            expected_shape = downscale_shape(expected_shape, scale)
            self.assertEqual(shape_level, expected_shape)

            effective_scale = [sc * eff for sc, eff in zip(scale, effective_scale)]
            if level > 0:
                self.assertEqual(effective_scale[::-1], ds.attrs["downsamplingFactors"])
            data = ds[:]
            self.assertFalse(np.allclose(data, 0))

    def check_result_bdv_hdf5(self, shape, scales):
        output_path = "./tmp/data.h5"
        with h5py.File(output_path, "r") as f:

            bdv_scale_factors = f["s00/resolutions"][:]

            expected_shape = shape
            effective_scale = [1, 1, 1]
            for level, scale in enumerate(scales):
                key = "t00000/s00/%i/cells" % level
                ds = f[key]
                shape_level = ds.shape
                expected_shape = downscale_shape(expected_shape, scale)
                self.assertEqual(shape_level, expected_shape)

                effective_scale = [sc * eff for sc, eff in zip(scale, effective_scale)]
                self.assertEqual(effective_scale[::-1], bdv_scale_factors[level].tolist())
            data = ds[:]
            self.assertFalse(np.allclose(data, 0))

    def check_result_bdv_n5(self, shape, scales):
        f = z5py.File(self.output_path)
        g = f["setup0"]
        bdv_scale_factors = g.attrs["downsamplingFactors"]
        bdv_dtype = np.dtype(g.attrs["dataType"])
        g = g["timepoint0"]
        self.assertTrue(g.attrs["multiScale"])
        self.assertIn("resolution", g.attrs)

        expected_shape = shape
        effective_scale = [1, 1, 1]
        for level, scale in enumerate(scales):
            key = "s%i" % level
            ds = g[key]
            self.assertEqual(np.dtype(ds.dtype), bdv_dtype)

            shape_level = ds.shape
            expected_shape = downscale_shape(expected_shape, scale)
            self.assertEqual(shape_level, expected_shape)

            effective_scale = [sc * eff for sc, eff in zip(scale, effective_scale)]
            if level > 0:
                self.assertEqual(effective_scale[::-1], bdv_scale_factors[level])
                self.assertEqual(effective_scale[::-1], ds.attrs["downsamplingFactors"])
            data = ds[:]
            self.assertFalse(np.allclose(data, 0))

    def check_result_ome_zarr(self, shape, scales):
        f = z5py.File("./tmp/data.ome.zarr")
        multiscales = f.attrs["multiscales"]
        self.assertEqual(len(multiscales), 1)
        multiscales = multiscales[0]
        datasets = multiscales["datasets"]
        expected_shape = shape
        for level, scale in enumerate(scales):
            key = datasets[level]["path"]
            ds = f[key]
            shape_level = ds.shape
            expected_shape = downscale_shape(expected_shape, scale)
            self.assertEqual(shape_level, expected_shape)
            data = ds[:]
            self.assertFalse(np.allclose(data, 0))

    def _downscale(self, metadata_format):
        from cluster_tools.downscaling import DownscalingWorkflow
        task = DownscalingWorkflow

        scales = [[1, 2, 2], [1, 2, 2], [2, 2, 2]]
        halos = [[1, 4, 4], [1, 4, 4], [2, 4, 4]]

        if metadata_format == "paintera":
            output_key_prefix = self.output_key_prefix
        else:
            output_key_prefix = ""

        max_jobs = self.max_jobs
        if metadata_format == "bdv.hdf5":
            output_path = "./tmp/data.h5"
            max_jobs = 1
        elif metadata_format == "ome.zarr":
            output_path = "./tmp/data.ome.zarr"
        else:
            output_path = self.output_path

        t = task(input_path=self.input_path, input_key=self.input_key,
                 output_path=output_path, output_key_prefix=output_key_prefix,
                 scale_factors=scales, halos=halos,
                 config_dir=self.config_folder, tmp_folder=self.tmp_folder,
                 target=self.target, max_jobs=max_jobs,
                 metadata_format=metadata_format)

        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)
        return scales

    def test_downscaling_paintera(self):
        scales = self._downscale(metadata_format="paintera")
        shape = z5py.File(self.input_path)[self.input_key].shape
        scales = [[1, 1, 1]] + scales
        self.check_result_paintera(shape, scales)

    def test_downscaling_bdv_h5(self):
        scales = self._downscale(metadata_format="bdv.hdf5")
        shape = z5py.File(self.input_path)[self.input_key].shape
        scales = [[1, 1, 1]] + scales
        self.check_result_bdv_hdf5(shape, scales)

    def test_downscaling_bdv_n5(self):
        scales = self._downscale(metadata_format="bdv.n5")
        shape = z5py.File(self.input_path)[self.input_key].shape
        scales = [[1, 1, 1]] + scales
        self.check_result_bdv_n5(shape, scales)

    def test_downscaling_ome_zarr(self):
        scales = self._downscale(metadata_format="ome.zarr")
        shape = z5py.File(self.input_path)[self.input_key].shape
        scales = [[1, 1, 1]] + scales
        self.check_result_ome_zarr(shape, scales)


if __name__ == "__main__":
    unittest.main()
