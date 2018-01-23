#! /usr/bin/python

import os
import argparse
import numpy as np
import nifty
import z5py


# Third pass:
# We assign all the nodes their final ids for each block
#
def assign_node_ids(block_id, blocking, ds_out, node_labeling):
    block = blocking.getBlock(block_id)
    bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
    subvol = ds_out[bb]
    # map node-ids in block
    # TODO this can be done faster !!!!!
    # maybe we should do this in nifty tools
    sub_ids = np.unique(subvol)
    sub_ids = sub_ids[sub_ids != 0]
    for sub_id in sub_ids:
        subvol[subvol == sub_id] = node_labeling[sub_id]
    ds_out[bb] = subvol


def cc_ufd_step4(block_file, out_path, out_key, tmp_folder, block_shape):

    ds_out = z5py.File(out_path, use_zarr_format=False)[out_key]
    shape = ds_out.shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    node_labeling = np.load(os.path.join(tmp_folder, 'node_labeling.npy'))

    block_list = np.load(block_file)
    # we get the job id from the file name
    job_id = int(os.path.split(block_file)[1].split('_')[2][:-4])

    for block_id in block_list:
        assign_node_ids(block_id, blocking, ds_out, node_labeling)

    print("Success job %i" % job_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('block_file', str)
    parser.add_argument('out_path', str)
    parser.add_argument('out_key', str)
    parser.add_argument('tmp_folder', str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    args = parser.parse_args()
    cc_ufd_step4(args.block_file, args.out_path,
                 args.out_key, args.tmp_folder,
                 list(args.block_shape))
