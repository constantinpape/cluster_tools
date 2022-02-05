import os
import sys
import unittest
import luigi
import numpy as np
import cluster_tools.utils.volume_utils as vu


try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestPixelClassification(BaseTest):
    input_key = "volumes/raw/s0"
    output_key = "ilastik_prediction"
    # TODO upload an ilastik project and use for proper tests
    pixel_classification_proj = "/home/pape/Work/my_projects/jils-project/ilastik_projects/jil/vol3_2D_pixelclass.ilp"

    def _test_ilastik_prediction(self, out_channels, ilp):
        from cluster_tools.ilastik import IlastikPredictionWorkflow
        halo = [2, 2, 2]
        task = IlastikPredictionWorkflow(
            tmp_folder=self.tmp_folder, config_dir=self.config_folder,
            target=self.target, max_jobs=self.max_jobs,
            input_path=self.input_path, input_key=self.input_key,
            output_path=self.output_path, output_key=self.output_key,
            halo=halo, ilastik_project=ilp, out_channels=out_channels
        )
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)

        self.assertTrue(os.path.exists(self.output_path))
        with vu.file_reader(self.output_path, "r") as f:
            self.assertTrue(self.output_key in f)
            pred = f[self.output_key][:]

        with vu.file_reader(self.input_path, "r") as f:
            shape = f[self.input_key].shape

        if out_channels is None:
            # we have 2 output channels in total
            exp_shape = (2,) + shape
        elif len(out_channels) == 1:
            exp_shape = shape
        else:
            exp_shape = (len(out_channels),) + shape
        self.assertEqual(pred.shape, exp_shape)
        self.assertFalse(np.allclose(pred, 0))

    def test_pixel_classification(self):
        self._test_ilastik_prediction(out_channels=None, ilp=self.pixel_classification_proj)

    def test_pixel_classification_out_channels(self):
        self._test_ilastik_prediction(out_channels=[0], ilp=self.pixel_classification_proj)


if __name__ == "__main__":
    unittest.main()
