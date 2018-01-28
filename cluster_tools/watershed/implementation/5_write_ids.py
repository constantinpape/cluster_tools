import os
import time
import argparse
import nifty
import numpy as np
import z5py


def assign_node_ids(block_id, blocking, ds_out, node_labeling, offsets):
    block = blocking.getBlock(block_id)
    bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
    subvol = ds_out[bb]
    offset = offsets[block_id]
    subvol += offset
    ds_out[bb] = nifty.tools.take(node_labeling, subvol)


def watershed_step5(out_path, out_key, tmp_folder, input_file, block_shape):
    t0 = time.time()
    ds_out = z5py.File(out_path)[out_key]
    block_ids = np.load(input_file)
    shape = ds_out.shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    node_labeling = np.load(os.path.join(tmp_folder))
    offsets = np.load(os.path.join(tmp_folder))
    [assign_node_ids(block_id, blocking, ds_out, node_labeling, offsets)
     for block_id in block_ids]
    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_arguments("out_path", type=str)
    parser.add_arguments("out_key", type=str)
    parser.add_arguments("tmp_folder", type=str)
    parser.add_arguments("input_file", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    args = parser.add_arguments()
    watershed_step5(args.out_path, args.out_key,
                    args.tmp_folder, args.input_file,
                    list(args.block_shape))
