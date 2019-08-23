import sys
import unittest

import numpy as np
import luigi
import z5py

from elf.skeleton import skeletonize
import elf.skeleton.io as skelio

try:
    from ..base import BaseTest
except ValueError:
    sys.path.append('..')
    from base import BaseTest


class TestSkeletons(BaseTest):
    seg_key = 'volumes/segmentation/multicut'
    out_key = 'skeletons'
    resolution = (1, 1, 1)
    size_thresh = 1000

    def ids_and_seg(self, n_ids):
        with z5py.File(self.input_path) as f:
            ds = f[self.seg_key]
            ds.n_threads = self.max_jobs
            seg = ds[:]

        # pick n_ids random ids
        ids, counts = np.unique(seg, return_counts=True)
        ids, counts = ids[1:], counts[1:]
        ids = ids[counts > self.size_thresh]
        ids = np.random.choice(ids, n_ids)
        return seg, ids

    def test_skeletons(self):
        from cluster_tools.skeletons import SkeletonWorkflow
        task = SkeletonWorkflow

        conf = task.get_config()['skeletonize']
        conf.update({'chunk_len': 50})

        t = task(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                 target=self.target, max_jobs=self.max_jobs,
                 input_path=self.input_path, input_key=self.seg_key,
                 output_path=self.output_path, output_key=self.out_key,
                 resolution=self.resolution, size_threshold=self.size_thresh)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)

        # check output for correctness
        seg, ids = self.ids_and_seg(100)
        ds = z5py.File(self.output_path)[self.out_key]
        for seg_id in ids:
            # read the result from file
            coords, edges = skelio.read_n5(ds, seg_id)

            # compute the expected result
            mask = seg == seg_id
            coords_exp, edges_exp = skeletonize(mask)

            self.assertEqual(coords.shape, coords_exp.shape)
            self.assertTrue(np.allclose(coords, coords_exp))

            self.assertEqual(edges.shape, edges_exp.shape)
            self.assertTrue(np.allclose(edges, edges_exp))


if __name__ == '__main__':
    unittest.main()
