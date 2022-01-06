import os
import sys
import unittest

import luigi
import numpy as np
import z5py

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestLiftedFeatureWorkflow(BaseTest):
    labels_key = "volumes/segmentation/groundtruth"

    def setUp(self):
        super().setUp()
        self.compute_graph()

    def check_result(self):
        with z5py.File(self.output_path) as f:
            uv_ids = f["lifted_nh"][:]
            costs = f["lifted_feats"][:]
        self.assertEqual(len(uv_ids), len(costs))
        self.assertFalse((uv_ids == 0).all())
        self.assertFalse(np.allclose(costs, 0))

    def test_workflow(self):
        from cluster_tools.lifted_features import LiftedFeaturesFromNodeLabelsWorkflow
        task = LiftedFeaturesFromNodeLabelsWorkflow
        ret = luigi.build([task(ws_path=self.input_path,
                                ws_key=self.ws_key,
                                labels_path=self.input_path,
                                labels_key=self.labels_key,
                                graph_path=self.output_path,
                                graph_key=self.graph_key,
                                output_path=self.output_path,
                                nh_out_key="lifted_nh",
                                feat_out_key="lifted_feats",
                                prefix="test",
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        self.check_result()


if __name__ == "__main__":
    unittest.main()
