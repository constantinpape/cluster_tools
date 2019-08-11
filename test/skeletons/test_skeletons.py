import os
import sys
import json
import unittest

import numpy as np
import luigi
import z5py
from skimage.morphology import skeletonize_3d
from skan import csr
from cluster_tools.utils import skeleton_utils as su

try:
    from ..base import BaseTest
except ImportError:
    sys.path.append('..')
    from base import BaseTest


class TestSkeletons(BaseTest):
    input_prefix = 'volumes/segmentation/multicut'
    output_prefix = 'skeletons'

    def _run_skel_wf(self, format_, max_jobs):
        from cluster_tools.skeletons import SkeletonWorkflow
        task = SkeletonWorkflow(tmp_folder=self.tmp_folder,
                                config_dir=self.config_folder,
                                target=self.target, max_jobs=max_jobs,
                                input_path=self.path, input_prefix=self.input_prefix,
                                output_path=self.output_path, output_prefix=self.output_prefix,
                                work_scale=0, skeleton_format=format_)
        ret = luigi.build([task], local_scheduler=True)
        self.assertTrue(ret)
        f = z5py.File(self.output_path)
        self.assertTrue(self.output_prefix in f)
        out_key = os.path.join(self.output_prefix, 's0')
        self.assertTrue(out_key in f)

    def ids_and_seg(self, n_ids=10, min_size=250):
        in_key = os.path.join(self.input_prefix, 's0')
        with z5py.File(self.path) as f:
            ds = f[in_key]
            seg = ds[:]

        # pick n_ids random ids
        ids, counts = np.unique(seg, return_counts=True)
        ids, counts = ids[1:], counts[1:]
        ids = ids[counts > min_size]
        ids = np.random.choice(ids, n_ids)
        return seg, ids

    def test_skeletons_n5(self):
        from cluster_tools.skeletons import SkeletonWorkflow
        config = SkeletonWorkflow.get_config()['skeletonize']
        config.update({'chunk_len': 50})
        with open(os.path.join(self.config_folder, 'skeletonize.config'), 'w') as f:
            json.dump(config, f)

        self._run_skel_wf(format_='n5', max_jobs=8)

        # check output for correctness
        seg, ids = self.ids_and_seg()
        out_key = os.path.join(self.output_prefix, 's0')
        ds = z5py.File(self.output_path)[out_key]
        for seg_id in ids:
            # read the result from file
            coords, edges = su.read_n5(ds, seg_id)

            # compute the expected result
            mask = seg == seg_id
            skel_vol = skeletonize_3d(mask)
            try:
                pix_graph, coords_exp, _ = csr.skeleton_to_csgraph(skel_vol)
            except ValueError:
                continue

            # check coordinates
            coords_exp = coords_exp[1:].astype('uint64')
            self.assertEqual(coords.shape, coords_exp.shape)
            self.assertTrue(np.allclose(coords, coords_exp))

            # check edges
            graph = csr.numba_csgraph(pix_graph)
            n_points = len(coords)
            edges_exp = [[u, v] for u in range(1, n_points + 1)
                         for v in graph.neighbors(u) if u < v]
            edges_exp = np.array(edges_exp)
            edges_exp -= 1
            self.assertEqual(edges.shape, edges_exp.shape)
            self.assertTrue(np.allclose(edges, edges_exp))

    def test_skeletons_swc(self):
        from cluster_tools.skeletons import SkeletonWorkflow
        config = SkeletonWorkflow.get_config()['skeletonize']
        config.update({'chunk_len': 50})
        with open(os.path.join(self.config_folder, 'skeletonize.config'), 'w') as f:
            json.dump(config, f)

        self._run_skel_wf(format_='swc', max_jobs=8)
        # check output for correctness
        seg, ids = self.ids_and_seg()
        out_folder = os.path.join(self.output_path, self.output_prefix, 's0')
        for seg_id in ids:
            # read the result from file
            out_file = os.path.join(out_folder, '%i.swc' % seg_id)
            skel_ids, coords, parents = su.read_swc(out_file)
            coords = np.array(coords, dtype='float')

            # compute the expected result
            mask = seg == seg_id
            skel_vol = skeletonize_3d(mask)
            try:
                pix_graph, coords_exp, _ = csr.skeleton_to_csgraph(skel_vol)
            except ValueError:
                continue

            # check coordinates
            coords_exp = coords_exp[1:]
            self.assertEqual(coords.shape, coords_exp.shape)
            self.assertTrue(np.allclose(coords, coords_exp))

            # TODO check parents

    def test_skeletons_volume(self):
        from cluster_tools.skeletons import SkeletonWorkflow
        config = SkeletonWorkflow.get_config()['skeletonize']
        config.update({'threads_per_job': 8})
        with open(os.path.join(self.config_folder, 'skeletonize.config'), 'w') as f:
            json.dump(config, f)

        self._run_skel_wf(format_='volume', max_jobs=1)

        # check output for correctness
        seg, ids = self.ids_and_seg()
        out_key = os.path.join(self.output_prefix, 's0')
        result = z5py.File(self.output_path)[out_key][:]
        for seg_id in ids:
            res = result == seg_id
            mask = seg == seg_id
            exp = skeletonize_3d(mask) > 0
            self.assertTrue(np.allclose(res, exp))


if __name__ == '__main__':
    unittest.main()
