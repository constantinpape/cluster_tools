import sys
import unittest
import numpy as np

import luigi
import z5py

try:
    from ..base import BaseTest
except ImportError:
    sys.path.append('..')
    from base import BaseTest


# TODO need labels for this !
class TestLiftedFeatureWorkflow(BaseTest):
    ws_key = 'volumes/watershed'
    labels_key = 'volumes/test_labels'
    graph_key = 'graph'

    def _check_result(self):
        out_path = self.tmp_folder + '/lifted_feats.n5'
        with z5py.File(out_path) as f:
            uv_ids = f['lifted_nh'][:]
            costs = f['lifted_feats'][:]
        self.assertEqual(len(uv_ids), len(costs))
        self.assertFalse((uv_ids == 0).all())
        self.assertFalse(np.allclose(costs, 0))

    def test_workflow(self):
        from cluster_tools.lifted_features import LiftedFeaturesFromNodeLabelsWorkflow
        out_path = self.tmp_folder + '/lifted_feats.n5'
        task = LiftedFeaturesFromNodeLabelsWorkflow
        ret = luigi.build([task(ws_path=self.input_path,
                                ws_key=self.ws_key,
                                labels_path=self.input_path,
                                labels_key=self.labels_key,
                                graph_path=self.input_path,
                                graph_key=self.graph_key,
                                output_path=out_path,
                                nh_out_key='lifted_nh',
                                feat_out_key='lifted_feats',
                                prefix='test',
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        self._check_result()


if __name__ == '__main__':
    unittest.main()
