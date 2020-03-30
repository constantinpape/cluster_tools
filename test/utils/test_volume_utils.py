import os
import unittest
from shutil import rmtree

import numpy as np


class TestVolumeUtils(unittest.TestCase):
    tmp_dir = './tmp'

    def setUp(self):
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def test_file_reader(self):
        from cluster_tools.utils.volume_utils import file_reader

        def _test_io(f):
            data = np.random.rand(100, 100)
            ds = f.create_dataset('data', data=data, chunks=(10, 10))
            out = ds[:]
            self.assertEqual(out.shape, data.shape)
            self.assertTrue(np.allclose(out, data))

        # test n5
        path = os.path.join(self.tmp_dir, 'a.n5')
        with file_reader(path) as f:
            _test_io(f)

        # test zr
        path = os.path.join(self.tmp_dir, 'a.zr')
        with file_reader(path) as f:
            _test_io(f)

        # test h5
        path = os.path.join(self.tmp_dir, 'a.h5')
        with file_reader(path) as f:
            _test_io(f)

    def test_blocks_in_volume(self):
        from cluster_tools.utils.volume_utils import blocks_in_volume, file_reader

        p = os.path.join(self.tmp_dir, 'data.n5')
        f = file_reader(p)

        def check_block_list(blocking, block_list, ds, bb=np.s_[:]):

            for block_id in block_list:
                block = blocking.getBlock(block_id)
                this_bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
                ds[this_bb] = block_id

            out = ds[bb]
            self.assertTrue((out > 0).all())

        # test vanilla
        k = 'd1'
        shape = (128,) * 3
        block_shape = (32,) * 3
        ds = f.create_dataset(k, shape=shape, chunks=block_shape, dtype='uint8')
        block_list, blocking = blocks_in_volume(shape, block_shape, return_blocking=True)
        check_block_list(blocking, block_list, ds)

        # test with roi
        k = 'd2'
        shape = (1023, 2049, 355)
        block_shape = (65, 93, 24)
        ds = f.create_dataset(k, shape=shape, chunks=block_shape, dtype='uint8')
        roi_begin = (104, 1039, 27)
        roi_end = (911, 1855, 134)
        block_list, blocking = blocks_in_volume(shape, block_shape, return_blocking=True,
                                                roi_begin=roi_begin, roi_end=roi_end)
        bb = tuple(slice(rb, re) for rb, re in zip(roi_begin, roi_end))
        check_block_list(blocking, block_list, ds, bb)


if __name__ == '__main__':
    unittest.main()
