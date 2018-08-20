#! /usr/bin/python

import os
import time
import argparse
import nifty
import numpy as np
import z5py


# TODO don't write empty output
def assign_node_ids(block_id, blocking, ds_out, node_labeling, offsets):
    block = blocking.getBlock(block_id)
    bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
    subvol = ds_out[bb]
    offset = offsets[block_id]
    # add offsets only for the non-masked part
    subvol[subvol != 0] += offset
    ds_out[bb] = nifty.tools.take(node_labeling, subvol).astype(subvol.dtype)


def masked_watershed_step5(out_path, out_key, tmp_folder, input_file, block_shape):
    t0 = time.time()
    ds_out = z5py.File(out_path)[out_key]
    block_ids = np.load(input_file)
    shape = ds_out.shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    node_labeling = np.load(os.path.join(tmp_folder, 'node_labeling.npy'))
    offsets = np.load(os.path.join(tmp_folder, 'offsets.npy'))
    [assign_node_ids(block_id, blocking, ds_out, node_labeling, offsets)
     for block_id in block_ids]
    job_id = int(os.path.split(input_file)[1].split('_')[2][:-4])

    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("tmp_folder", type=str)
    parser.add_argument("input_file", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    args = parser.parse_args()
    masked_watershed_step5(args.out_path, args.out_key,
                           args.tmp_folder, args.input_file,
                           list(args.block_shape))
