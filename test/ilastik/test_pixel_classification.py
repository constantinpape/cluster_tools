import os
import sys
import unittest
import luigi
import numpy as np
import cluster_tools.utils.volume_utils as vu

try:
    from ilastik.experimental.api import from_project_file
except Exception:
    from_project_file = None

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


@unittest.skipIf(from_project_file is None, "Need ilastik api")
class TestPixelClassification(BaseTest):
    input_key = "volumes/raw/s0"
    output_key = "ilastik_prediction"
    mask_key = "volumes/mask"
    pixel_classification_proj = os.path.join(os.path.split(__file__)[0], "../../test_data/cremi-test-project.ilp")

    def _test_ilastik_prediction(self, out_channels, ilp, mask_path="", mask_key=""):
        from cluster_tools.ilastik import IlastikPredictionWorkflow
        halo = [2, 2, 2]
        task = IlastikPredictionWorkflow(
            tmp_folder=self.tmp_folder, config_dir=self.config_folder,
            target=self.target, max_jobs=self.max_jobs,
            input_path=self.input_path, input_key=self.input_key,
            output_path=self.output_path, output_key=self.output_key,
            halo=halo, ilastik_project=ilp, out_channels=out_channels,
            mask_path=mask_path, mask_key=mask_key
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

        # TODO check if the mask is actually 0 once prediction masking is included in ilastik api
        if mask_path != "":
            pass

    def test_pixel_classification(self):
        self._test_ilastik_prediction(out_channels=None, ilp=self.pixel_classification_proj)

    def test_pixel_classification_out_channels(self):
        self._test_ilastik_prediction(out_channels=[0], ilp=self.pixel_classification_proj)

    def test_pixel_classification_with_mask(self):
        self._test_ilastik_prediction(out_channels=None, ilp=self.pixel_classification_proj,
                                      mask_path=self.input_path, mask_key=self.mask_key)


if __name__ == "__main__":
    unittest.main()
