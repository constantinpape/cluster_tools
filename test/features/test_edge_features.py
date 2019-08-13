import json
import os
import unittest
import sys
from functools import partial

import numpy as np
import luigi
import z5py

import nifty.tools as nt
import nifty.graph.rag as nrag
from cluster_tools.utils.volume_utils import normalize

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestEdgeFeatures(BaseTest):
    output_key = 'features'
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

    def setUp(self):
        super().setUp()
        self.compute_graph()

    def _check_subresults(self, in_key, feat_func):
        f = z5py.File(self.input_path)
        f_feat = z5py.File(self.output_path)
        ds_inp = f[in_key]
        ds_ws = f[self.ws_key]

        shape = ds_ws.shape
        blocking = nt.blocking([0, 0, 0], list(shape),
                               self.block_shape)

        halo = [1, 1, 1]
        for block_id in range(blocking.numberOfBlocks):

            # get the block with the appropriate halo
            # and the corresponding bounding box
            block = blocking.getBlockWithHalo(block_id, halo)
            outer_block, inner_block = block.outerBlock, block.innerBlock
            bb = tuple(slice(beg, end) for beg, end in zip(inner_block.begin,
                                                           outer_block.end))
            # load the segmentation and the input
            # and compute the nifty graph
            seg = ds_ws[bb]
            rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)
            inp = normalize(ds_inp[bb])

            # compute nifty features
            features_nifty = feat_func(rag, inp, 0., 1.)

            # load the features
            feat_key = os.path.join('blocks', 'block_%i' % block_id)
            features_block = f_feat[feat_key][:]

            # compare features
            self.assertEqual(len(features_nifty), len(features_block))
            self.assertEqual(features_nifty.shape[1], features_block.shape[1] - 1)
            self.assertTrue(np.allclose(features_nifty, features_block[:, :-1]))
            len_nifty = nrag.accumulateEdgeMeanAndLength(rag, inp)[:, 1]
            self.assertTrue(np.allclose(len_nifty, features_block[:, -1]))

    def _check_fullresults(self, in_key, feat_func):
        f = z5py.File(self.input_path)
        ds_inp = f[in_key]
        ds_inp.n_threads = 8
        ds_ws = f[self.ws_key]
        ds_ws.n_threads = 8

        seg = ds_ws[:]
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)
        inp = ds_inp[:]

        # compute nifty features
        features_nifty = feat_func(rag, inp, 0., 1.)
        # load features
        features = z5py.File(self.output_path)[self.output_key][:]
        self.assertEqual(len(features_nifty), len(features))
        self.assertEqual(features_nifty.shape[1], features.shape[1] - 1)

        # we can only assert equality for mean, std, min, max and len
        print(features_nifty[:10, 0])
        print(features[:10, 0])
        # -> mean
        self.assertTrue(np.allclose(features_nifty[:, 0], features[:, 0]))
        # -> std
        self.assertTrue(np.allclose(features_nifty[:, 1], features[:, 1]))
        # -> min
        self.assertTrue(np.allclose(features_nifty[:, 2], features[:, 2]))
        # -> max
        self.assertTrue(np.allclose(features_nifty[:, 8], features[:, 8]))
        # self.assertFalse(np.allcose(features[:, 3:8], 0))
        # check that the edge-lens agree
        # len_nifty = nrag.accumulateEdgeMeanAndLength(rag, inp)[:, 1]
        # self.assertTrue(np.allclose(len_nifty, features_block[:, -1]))

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
        self._check_subresults(self.boundary_key,
                               nrag.accumulateEdgeStandartFeatures)
        self._check_fullresults(self.boundary_key,
                                nrag.accumulateEdgeStandartFeatures)

    def test_affinity_features(self):
        from cluster_tools.features import EdgeFeaturesWorkflow
        task = EdgeFeaturesWorkflow

        config = task.get_config()['block_edge_features']
        config.update({'offsets': self.offsets})
        with open(os.path.join(self.config_folder, 'block_edge_features.json'), 'w') as f:
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
                            offsets=self.offsets)
        self._check_subresults(self.aff_key, feat_func)
        self._check_fullresults(self.aff_key, feat_func)


if __name__ == '__main__':
    unittest.main()
