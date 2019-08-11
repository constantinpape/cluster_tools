import os
import sys
import json
import unittest
from shutil import rmtree

import z5py
import luigi

try:
    from cluster_tools.morphology import LabelMultisetWorkflow
except ImportError:
    sys.path.append('../..')
    from cluster_tools.morphology import LabelMultisetWorkflow


class TestLabelMultisets(unittest.TestCase):
    input_path = '/g/kreshuk/pape/Work/data/cremi/sampleA+.h5'
    input_key = 'segmentation/multicut'

    expected_path = '/g/kreshuk/pape/Work/data/cremi/sampleA+_paintera.n5'
    expected_key = 'segmentation/multicut/data'

    output_path = './tmp/data.n5'
    output_key = 'data'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    target = 'local'

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)
        os.makedirs(self.config_folder, exist_ok=True)
        self.config = LabelMultisetWorkflow.get_config()
        global_config = self.config['global']
        global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
        global_config['block_shape'] = []  # TODO
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_label_multisets(self):
        task = LabelMultisetWorkflow

        # set scale factors and restrict sets same as paintera helper:
        # paintera-conversion-helper -r -d sampleA+.h5,raw,raw -d sampleA+.h5,segmentation/multicut,label
        #                            -o sampleA+_paintera.n5 -b 64,64,64
        #                            -s 2,2,1 2,2,1 2,2,1, 2,2,2 -m -1 -1 5 3
        scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]]
        restrict_sets = [-1, -1, 5, 3]
        t = task(tmp_folder=self.tmp_folder, max_jobs=4,
                 config_dir=self.config_folder, target=self.target,
                 input_path=self.input_path, input_key=self.input_key,
                 output_path=self.output_path, output_prefix=self.output_key,
                 scale_factors=scale_factors, restrict_sets=restrict_sets)
        ret = luigi.build([t], local_scheduler=True)
        self.assertTrue(ret)

        f = z5py.File(self.output_path)
        g = f[self.output_key]

        f_exp = z5py.File(self.expected_path)
        g_exp = f_exp[self.expected_key]

        # check all scales
        for scale in g:
            self.assertTrue(scale in g_exp)
            ds = g[scale]
            ds_exp = g_exp[scale]
            self.assertEqual(ds.shape, ds_exp.shape)
            self.assertEqual(ds.chunks, ds_exp.chunks)
            chunks_per_dim = ds.chunks_per_dimension
            for z in range(chunks_per_dim[0]):
                for y in range(chunks_per_dim[1]):
                    for x in range(chunks_per_dim[2]):
                        chunk_id = (z, y, x)
                        out = ds.read_chunk(chunk_id)
                        out_exp = ds_exp.read_chunk(chunk_id)
                        if out is None:
                            self.assertTrue(out_exp is None)
                            continue
                        self.assertEqual(out.shape, out_exp.shape)
                        self.assertTrue(np.allclose(out, out_exp))



if __name__ == '__main__':
    unittest.main()
