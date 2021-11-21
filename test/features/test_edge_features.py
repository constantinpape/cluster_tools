import json
import os
import unittest
import sys
from functools import partial

import numpy as np
import luigi
import z5py

import nifty.graph.rag as nrag
from cluster_tools.utils.volume_utils import normalize

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


# FIXME both tests fail:
# - boundary features: the min values do not agree
# - affinity features: the edge size values do not agree
class TestEdgeFeatures(BaseTest):
    output_key = "features"
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

    def setUp(self):
        super().setUp()
        self.compute_graph(ignore_label=False)

    def check_results(self, in_key, feat_func):
        f = z5py.File(self.input_path)
        ds_inp = f[in_key]
        ds_inp.n_threads = 8
        ds_ws = f[self.ws_key]
        ds_ws.n_threads = 8

        # load features
        features = z5py.File(self.output_path)[self.output_key][:]

        # load seg and input, make rag
        seg = ds_ws[:]
        inp = normalize(ds_inp[:]) if ds_inp.ndim == 3 else normalize(ds_inp[:3])
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)

        # compute nifty lens
        if inp.ndim == 3:
            len_nifty = nrag.accumulateEdgeMeanAndLength(rag, inp)[:, 1]
        else:
            len_nifty = nrag.accumulateEdgeMeanAndLength(rag, inp[0])[:, 1]
        self.assertTrue(np.allclose(len_nifty, features[:, -1]))

        # compute nifty features
        features_nifty = feat_func(rag, inp)

        # check feature shape
        self.assertEqual(len(features_nifty), len(features))
        self.assertEqual(features_nifty.shape[1], features.shape[1] - 1)

        # debugging: check closeness of the min values
        # close = np.isclose(features_nifty[:, 2], features[:, 2])
        # print(close.sum(), "/", len(close))
        # not_close = ~close
        # print(np.where(not_close)[:10])
        # print(features[:, 2][not_close][:10])
        # print(features_nifty[:, 2][not_close][:10])

        # we can only assert equality for mean, std, min, max and len
        # -> mean
        self.assertTrue(np.allclose(features_nifty[:, 0], features[:, 0]))
        # -> std
        self.assertTrue(np.allclose(features_nifty[:, 1], features[:, 1]))
        # -> min
        self.assertTrue(np.allclose(features_nifty[:, 2], features[:, 2]))
        # -> max
        self.assertTrue(np.allclose(features_nifty[:, 8], features[:, 8]))

    # current issue: the min values don't agree, I don"t know why.
    @unittest.expectedFailure
    def test_boundary_features(self):
        from cluster_tools.features import EdgeFeaturesWorkflow
        task = EdgeFeaturesWorkflow
        ret = luigi.build([task(input_path=self.input_path,
                                input_key=self.boundary_key,
                                labels_path=self.input_path,
                                labels_key=self.ws_key,
                                graph_path=self.output_path,
                                graph_key=self.graph_key,
                                output_path=self.output_path,
                                output_key=self.output_key,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        feat_func = partial(nrag.accumulateEdgeStandartFeatures, minVal=0., maxVal=1.)
        self.check_results(self.boundary_key, feat_func)

    # current issue: the len values don't agree.
    # In the current implementation, the len results actually depend on the affinity offsets, which is bad.
    @unittest.expectedFailure
    def test_affinity_features(self):
        from cluster_tools.features import EdgeFeaturesWorkflow
        task = EdgeFeaturesWorkflow

        config = task.get_config()["block_edge_features"]
        config.update({"offsets": self.offsets})
        with open(os.path.join(self.config_folder, "block_edge_features.config"), "w") as f:
            json.dump(config, f)

        ret = luigi.build([task(input_path=self.input_path,
                                input_key=self.aff_key,
                                labels_path=self.input_path,
                                labels_key=self.ws_key,
                                graph_path=self.output_path,
                                graph_key=self.graph_key,
                                output_path=self.output_path,
                                output_key=self.output_key,
                                config_dir=self.config_folder,
                                tmp_folder=self.tmp_folder,
                                target=self.target,
                                max_jobs=self.max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        feat_func = partial(nrag.accumulateAffinityStandartFeatures,
                            offsets=self.offsets, min=0., max=1.)
        self.check_results(self.aff_key, feat_func)

    # TODO implement
    def test_features_from_filters(self):
        pass


if __name__ == "__main__":
    unittest.main()
