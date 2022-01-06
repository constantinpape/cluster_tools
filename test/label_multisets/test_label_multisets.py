import os
import sys
import unittest

import nifty.tools as nt
import numpy as np
import luigi
from tqdm import trange
import z5py
from elf.label_multiset import deserialize_multiset

try:
    from ..base import BaseTest
except Exception:
    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
    from base import BaseTest


class TestLabelMultisets(BaseTest):
    """ Test for the label multiset converter.

    The expected data needs to be computed with 'paintera-conversion-helper'
    (conda install -c conda-forge paintera):
    paintera-conversion-helper -r -d sampleA.n5,volumes/raw/s0,raw
                               -d sampleA.n5,volumes/segmentation/multicut,label
                               -o sampleA_paintera.n5 -b 256,256,32
                               -s 2,2,1 2,2,1 2,2,1, 2,2,2 -m -1 -1 5 3
    """
    input_key = "volumes/segmentation/multicut"
    output_key = "data"
    expected_key = "volumes/segmentation/label_multiset/data"

    def check_serialization(self, res, exp):
        self.assertEqual(res.shape, exp.shape)
        if not np.array_equal(res, exp):
            return False
        return True

    def check_multisets(self, res, exp):
        # check number of sets and entries
        self.assertEqual(res.size, exp.size)
        self.assertEqual(res.n_entries, exp.n_entries)
        # check amax vector
        self.assertTrue(np.array_equal(res.argmax, exp.argmax))
        # check offset vector
        if not np.array_equal(res.offsets, exp.offsets):
            return False
        # check ids and counts
        if not np.array_equal(res.ids, exp.ids) or not np.array_equal(res.counts, exp.counts):
            return False
        return True

    def check_pixels(self, res, exp):
        # only check 4 cubed blocks to save time
        blocking = nt.blocking([0, 0, 0], res.shape, [4, 4, 4])
        for block_id in trange(blocking.numberOfBlocks):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            ids_res, counts_res = res[bb]
            ids_exp, counts_exp = exp[bb]
            self.assertTrue(np.array_equal(ids_res, ids_exp))
            self.assertTrue(np.array_equal(counts_res, counts_exp))

    def check_chunk(self, res, exp, shape):
        # 1.) check direct serialization, if it matches, its fine
        # otherwise, we need to continue with checks
        # (failures might be due to permutation invariance)
        match = self.check_serialization(res, exp)
        if match:
            return

        # 2.) check multi-set members
        res = deserialize_multiset(res, shape)
        exp = deserialize_multiset(exp, shape)
        match = self.check_multisets(res, exp)
        if match:
            return

        # 3.) check pixel-wise agreement
        self.check_pixels(res, exp)

    def check_empty(self, res, shape):
        res = deserialize_multiset(res, shape)
        self.assertEqual(res.n_entries, 1)
        self.assertTrue(np.array_equal(res.ids, np.array([0], dtype="uint64")))

    def test_label_multisets(self):
        from cluster_tools.label_multisets import LabelMultisetWorkflow
        task = LabelMultisetWorkflow

        scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]]
        restrict_sets = [-1, -1, 5, 3]
        t = task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                 config_dir=self.config_folder, target=self.target,
                 input_path=self.input_path, input_key=self.input_key,
                 output_path=self.output_path, output_prefix=self.output_key,
                 scale_factors=scale_factors, restrict_sets=restrict_sets)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)

        f = z5py.File(self.output_path, "r")
        f_exp = z5py.File(self.input_path, "r")
        g = f[self.output_key]
        g_exp = f_exp[self.expected_key]

        # check all scales
        for scale in g:
            print("checking scale:", scale)
            # We skip the test for s4 because it fails due to an inconsistency
            # in the implementations,
            # see https://github.com/saalfeldlab/imglib2-label-multisets/issues/14
            if scale == "s4":
                continue
            self.assertTrue(scale in g_exp)
            ds = g[scale]
            ds_exp = g_exp[scale]
            self.assertEqual(ds.shape, ds_exp.shape)
            self.assertEqual(ds.chunks, ds_exp.chunks)

            # check the metadata
            attrs = ds.attrs

            self.assertTrue(attrs["isLabelMultiset"])

            scale_factor = attrs.get("downsamplingFactors", None)
            attrs_exp = ds_exp.attrs
            scale_factor_exp = attrs_exp.get("downsamplingFactors", None)
            self.assertEqual(scale_factor, scale_factor_exp)

            restrict = attrs.get("maxNumEntries", -1)
            restrict_exp = attrs_exp.get("maxNumEntries", -1)
            self.assertEqual(restrict, restrict_exp)

            mid = attrs.get("maxId", None)
            mid_exp = attrs_exp.get("maxId", None)
            self.assertEqual(mid, mid_exp)

            blocking = nt.blocking([0, 0, 0], ds.shape, ds.chunks)
            for block_id in range(blocking.numberOfBlocks):
                block = blocking.getBlock(block_id)
                chunk_id = tuple(beg // ch for beg, ch in zip(block.begin, ds.chunks))
                chunk_shape = block.shape
                out = ds.read_chunk(chunk_id)
                out_exp = ds_exp.read_chunk(chunk_id)
                print("Checking chunk:", chunk_id)
                if out is None:
                    # NOTE paintera-conversion currently is writing out empty chunks.
                    # this might change in the future and then we can just check that `out_exp is None`
                    # For now, we have to check that the chunk is empty
                    # self.assertTrue(out_exp is None)
                    self.check_empty(out_exp, chunk_shape)
                else:
                    self.check_chunk(out, out_exp, chunk_shape)


if __name__ == "__main__":
    unittest.main()
