import os
import sys
import unittest

import numpy as np
import luigi
import z5py

import nifty.ground_truth as ngt
import nifty.distributed as ndist
import nifty.tools as nt

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestNodeLabels(BaseTest):
    input_key = 'volumes/segmentation/groundtruth'
    output_key = 'labels'

    @staticmethod
    def compute_overlaps(seg_a, seg_b, max_overlap=True):

        seg_ids = np.unique(seg_a)
        comp = ngt.overlap(seg_a, seg_b)
        overlaps = [comp.overlapArrays(ida, True) for ida in seg_ids]

        if max_overlap:
            # the max overlap can be ambiguous, need to filtr for this
            mask = np.array([ovlp[1][0] != ovlp[1][1] if ovlp[1].size > 1 else True
                             for ovlp in overlaps])
            overlaps = np.array([ovlp[0][0] for ovlp in overlaps])
            assert mask.shape == overlaps.shape
            return overlaps, mask
        else:
            overlaps = {seg_id: ovlp for seg_id, ovlp in zip(seg_ids, overlaps)}
            return overlaps

    def load_data(self):
        # compute the expected max overlaps
        with z5py.File(self.input_path) as f:
            ds_ws = f[self.ws_key]
            ds_ws.n_threads = self.max_jobs
            ws = ds_ws[:]

            ds_inp = f[self.input_key]
            ds_inp.n_threads = self.max_jobs
            inp = ds_inp[:]
        return ws, inp

    def check_overlaps(self, ids, overlaps, overlaps_exp):
        self.assertEqual(len(ids), len(overlaps))
        self.assertEqual(len(overlaps), len(overlaps_exp))

        for seg_id in ids:

            this_ovlps = overlaps[seg_id]
            ovlp_ids = np.array(list(this_ovlps.keys()))
            ovlp_counts = np.array(list(this_ovlps.values()))
            sorted_ids = np.argsort(ovlp_ids)
            ovlp_ids = ovlp_ids[sorted_ids]
            ovlp_counts = ovlp_counts[sorted_ids]

            ovlp_ids_exp, ovlp_counts_exp = overlaps_exp[seg_id]
            sorted_ids = np.argsort(ovlp_ids_exp)
            ovlp_ids_exp = ovlp_ids_exp[sorted_ids]
            ovlp_counts_exp = ovlp_counts_exp[sorted_ids]

            self.assertTrue(np.allclose(ovlp_ids, ovlp_ids_exp))
            self.assertTrue(np.allclose(ovlp_counts, ovlp_counts_exp))

    def test_max_overlap(self):
        from cluster_tools.node_labels import NodeLabelWorkflow
        task = NodeLabelWorkflow(tmp_folder=self.tmp_folder,
                                 config_dir=self.config_folder,
                                 target=self.target, max_jobs=self.max_jobs,
                                 ws_path=self.input_path, ws_key=self.ws_key,
                                 input_path=self.input_path, input_key=self.input_key,
                                 output_path=self.output_path, output_key=self.output_key)

        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)

        # load the result
        with z5py.File(self.output_path) as f:
            overlaps = f[self.output_key][:]

        ws, inp = self.load_data()

        overlaps_exp, mask = self.compute_overlaps(ws, inp)
        self.assertEqual(overlaps.shape, overlaps_exp.shape)

        overlaps = overlaps[mask]
        overlaps_exp = overlaps_exp[mask]

        # compare results
        self.assertTrue(np.allclose(overlaps, overlaps_exp))

    def test_subresults(self):
        from cluster_tools.node_labels import NodeLabelWorkflow
        task = NodeLabelWorkflow(tmp_folder=self.tmp_folder,
                                 config_dir=self.config_folder,
                                 target=self.target, max_jobs=self.max_jobs,
                                 ws_path=self.input_path, ws_key=self.ws_key,
                                 input_path=self.input_path, input_key=self.input_key,
                                 output_path=self.output_path, output_key=self.output_key)

        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)

        tmp_path = os.path.join(self.output_path, 'label_overlaps_')
        ws, inp = self.load_data()

        blocking = nt.blocking([0, 0, 0], ws.shape, self.block_shape)
        for block_id in range(blocking.numberOfBlocks):
            block = blocking.getBlock(block_id)
            chunk_id = tuple(start // bs for start, bs in zip(block.begin, self.block_shape))
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

            wsb, inpb = ws[bb], inp[bb]

            overlaps, _ = ndist.deserializeOverlapChunk(tmp_path, chunk_id)
            overlaps_exp = self.compute_overlaps(wsb, inpb, False)

            ids = np.unique(wsb)
            self.check_overlaps(ids, overlaps, overlaps_exp)

    def test_overlaps(self):
        from cluster_tools.node_labels import NodeLabelWorkflow
        task = NodeLabelWorkflow(tmp_folder=self.tmp_folder,
                                 config_dir=self.config_folder,
                                 target=self.target, max_jobs=self.max_jobs,
                                 ws_path=self.input_path, ws_key=self.ws_key,
                                 input_path=self.input_path, input_key=self.input_key,
                                 output_path=self.output_path, output_key=self.output_key,
                                 max_overlap=False)

        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)

        # load the result
        overlaps = ndist.deserializeOverlapChunk(os.path.join(self.output_path,
                                                              self.output_key),
                                                 (0,))[0]

        # compute the expected overlaps
        ws, inp = self.load_data()
        overlaps_exp = self.compute_overlaps(ws, inp, max_overlap=False)

        # check the result
        ids = np.unique(ws)
        self.check_overlaps(ids, overlaps, overlaps_exp)


if __name__ == '__main__':
    unittest.main()
