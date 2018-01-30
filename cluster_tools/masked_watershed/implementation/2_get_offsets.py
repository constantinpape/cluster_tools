#! /usr/bin/python

import os
import time
import argparse
import numpy as np
import nifty
import z5py


def masked_watershed_step2(out_path, key_out, tmp_folder, block_shape):
    shape = z5py.File(out_path)[key_out].shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=shape,
                                    blockShape=block_shape)
    t0 = time.time()
    offsets = np.array([np.load(os.path.join(tmp_folder, '1_output_maxid_%i.npy' % block_id))
                        for block_id in range(blocking.numberOfBlocks)], dtype='uint64')

    last_max_id = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    max_id = offsets[-1] + last_max_id

    np.save(os.path.join(tmp_folder, 'max_id.npy'), max_id)
    np.save(os.path.join(tmp_folder, 'offsets.npy'), offsets)

    print("Success")
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)

    args = parser.parse_args()
    masked_watershed_step2(args.out_path, args.out_key,
                           args.tmp_folder, list(args.block_shape))
