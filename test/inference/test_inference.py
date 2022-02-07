import json
import os
import sys
import unittest
import luigi
import numpy as np
import cluster_tools.utils.volume_utils as vu

try:
    import bioimageio.core
    have_bioimageio = True
except ImportError:
    have_bioimageio = False

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestInference(BaseTest):
    input_key = "volumes/raw/s0"
    output_key = "prediction"
    model_doi = "10.5281/zenodo.5874741"

    def _test_inference(self, ckpt, framework):
        from cluster_tools.inference import InferenceLocal

        # set block shape that fits into the network
        block_shape = [14, 280, 280]
        halo = [1, 4, 4]
        conf_dir = self.config_folder
        conf_path = os.path.join(conf_dir, "global.config")
        with open(conf_path, "r") as f:
            config = json.load(f)
        config["block_shape"] = block_shape
        with open(conf_path, "w") as f:
            json.dump(config, f)

        out_key = {self.output_key: [0, 1]}
        task = InferenceLocal(
            tmp_folder=self.tmp_folder, config_dir=conf_dir, max_jobs=1,
            input_path=self.input_path, input_key=self.input_key,
            output_path=self.output_path, output_key=out_key,
            checkpoint_path=ckpt,  halo=halo, framework=framework
        )
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)

        self.assertTrue(os.path.exists(self.output_path))
        with vu.file_reader(self.output_path, "r") as f:
            self.assertTrue(self.output_key in f)
            pred = f[self.output_key][:]

        with vu.file_reader(self.input_path, "r") as f:
            exp_shape = f[self.input_key].shape
        self.assertEqual(pred.shape, exp_shape)
        self.assertFalse(np.allclose(pred, 0))

    @unittest.skipUnless(have_bioimageio is None, "Need bioimageio.core")
    def test_bioimageio(self):
        self._test_inference(self.model_doi, "bioimageio")

    # TODO extract the model weights from the example model
    # def test_pytorch(self)


if __name__ == "__main__":
    unittest.main()
