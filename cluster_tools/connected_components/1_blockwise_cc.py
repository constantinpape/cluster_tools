import os
import argparse
import json

import numpy as np
import vigra
import nifty
import z5py


def process_single_block(block_id, blocking, ds_in, ds_out, tmp_folder):
    shape = ds_in.shape
    halo = [1, 1, 1]

    block = blocking.getBlockWithHalo(block_id, halo)
    inner_block, outer_block, local_block = block.innerBlock, block.outerBlock, block.innerBlockLocal

    # we offset with the coordinate of the leftmost pixel
    offset = sum(e * s for e, s in zip(inner_block.begin, shape))

    # get all bounding boxes
    bb_outer = tuple(slice(b, e) for b, e in zip(outer_block.begin, outer_block.end))
    begin, end = inner_block.begin, inner_block.end
    bb_inner = tuple(slice(b, e) for b, e in zip(begin, end))
    bb_local = tuple(slice(b, e) for b, e in zip(local_block.begin, local_block.end))
    outer_shape = outer_block.shape

    # get the subvolume, find connected components and write non-overlapping part to file
    subvolume = ds_in[bb_outer]
    cc = vigra.analysis.labelVolumeWithBackground(subvolume).astype('uint64')
    cc[cc != 0] += offset
    ds_out[bb_inner] = cc[bb_local]

    # serialize all the overlaps
    overlap_ids = []
    for ii in range(6):
        axis = ii // 2
        to_lower = ii % 2
        neighbor_id = blocking.getNeighborId(block_id, axis=axis, lower=to_lower)

        if neighbor_id != -1:
            overlap_bb = tuple(slice(None) if i != axis else
                               slice(0, 2) if to_lower else
                               slice(outer_shape[i] - 2, outer_shape[i]) for i in range(3))

            overlap = cc[overlap_bb]

            vigra.writeHDF5(overlap, os.path.join(tmp_folder, 'block_ovlp_%i_%i.h5' % (block_id, neighbor_id)),
                            'data', compression='gzip')

            # we only return the overlap ids, if the block id is smaller than the neighbor id,
            # to keep the pairs unique
            if block_id < neighbor_id:
                overlap_ids.append((block_id, neighbor_id))
    max_id = int(cc.max())
    return overlap_ids, max_id


def cc_ufd_step1(in_path, in_key,
                 out_path, out_key,
                 block_shape, tmp_folder,
                 block_file):

    assert os.path.exists(in_path)
    assert os.path.exists(out_path)
    assert os.path.exists(tmp_folder)

    with open(block_file, 'r') as f:
        inputs = json.load(f)
    job_id = inputs["job_id"]
    block_list = inputs["block_list"]

    ds_in = z5py.File(in_path, use_zarr_format=False)[in_key]
    ds_out = z5py.File(out_path, use_zarr_format=False)[out_key]
    shape = ds_in.shape
    assert ds_out.shape == shape

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    result = [process_single_block(block_id, blocking, ds_in, ds_out, tmp_folder) for block_id in block_list]
    overlap_ids = [ids for res in result for ids in res[0]]
    max_id = np.max([res[1] for res in result])
    with open(os.path.join(tmp_folder, '1_output_%i.json' % job_id), 'w') as f:
        json.dump({"overlap_ids": overlap_ids, "max_id": max_id}, f)
    print("Success job %i" % job_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str)
    parser.add_argument("in_key", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--tmp_folder", str)
    parser.add_argument("--block_file", str)

    args = parser.parse_args()
    cc_ufd_step1(args.in_path, args.in_key,
                 args.out_path, args.out_key,
                 list(args.block_shape), args.tmp_folder,
                 args.block_file)
