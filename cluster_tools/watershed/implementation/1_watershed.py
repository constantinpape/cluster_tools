#! /usr/bin/python

import os
import argparse
import time
import numpy as np
import h5py

import vigra
import z5py
import nifty

import cremi_tools.segmentation as cseg


def find_overlaps(block_id, blocking, ws, inner_block, outer_block, local_block, halo, tmp_folder):
    # serialize the overlaps
    overlap_ids = []
    for ii in range(6):
        axis = ii // 2
        to_lower = ii % 2
        neighbor_id = blocking.getNeighborId(block_id, axis=axis, lower=to_lower)

        if neighbor_id != -1:
            overlap_bb = tuple(slice(None) if i != axis else
                               slice(0, 2*halo[i]) if to_lower else
                               slice(inner_block.end[i] - halo[i] - outer_block.begin[i],
                                     outer_block.end[i] - outer_block.begin[i]) for i in range(3))

            overlap = ws[overlap_bb]

            ovlp_path = os.path.join(tmp_folder, 'block_%i_%i.h5' % (block_id, neighbor_id))
            vigra.writeHDF5(overlap, ovlp_path, 'data', compression='gzip')

            with h5py.File(ovlp_path) as f:
                attrs = f['data'].attrs
                attrs['overlap_dimension'] = axis
                attrs['overlap_begin'] = tuple(local_block.begin[i] for i in range(3))
                attrs['overlap_end'] = tuple(local_block.end[i] for i in range(3))

            # we only return the overlap ids, if the block id is smaller than the neighbor id,
            # to keep the pairs unique
            if block_id < neighbor_id:
                overlap_ids.append((block_id, neighbor_id))
    return overlap_ids


def single_block_watershed(block_id, blocking,
                           ds_affs, ds_out,
                           halo, tmp_folder, segmenter):
    block = blocking.getBlockWithHalo(block_id, halo)

    # get all blockings
    outer_block, inner_block, local_block = block.outerBlock, block.innerBlock, block.innerBlockLocal
    inner_bb = tuple(slice(b, e) for b, e in zip(inner_block.begin, inner_block.end))
    outer_bb = tuple(slice(b, e) for b, e in zip(outer_block.begin, outer_block.end))
    local_bb = tuple(slice(b, e) for b, e in zip(local_block.begin, local_block.end))

    # load the data and average it for z affinities
    aff_bb = (slice(None),) + outer_bb
    affs = ds_affs[aff_bb]

    # run watershed
    ws, max_id = segmenter(affs)

    # save watershed in the inner (non-overlapping) block
    ds_out[inner_bb] = ws[local_bb].astype('uint64')

    overlap_ids = find_overlaps(block_id, blocking, ws,
                                inner_block, outer_block, local_block,
                                halo, tmp_folder)

    # serialize the max ids
    np.save(os.path.join(tmp_folder, '1_output_maxid_%i.npy' % block_id), max_id + 1)
    return overlap_ids


# check again hat distance ransform thresholds make sense
def watershed_step1(aff_path, aff_key, out_path, key_out,
                    out_blocks, tmp_folder, block_file, halo=[5, 50, 50],
                    threshold_cc=.95, threshold_dt=.5, sigma_seeds=1.,
                    invert_input=True):

    t0 = time.time()
    ds_affs = z5py.File(aff_path)[aff_key]
    ds_out = z5py.File(out_path, use_zarr_format=False)[key_out]
    shape = ds_affs.shape[1:]
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(out_blocks))
    block_list = np.load(block_file)

    # the segmenter
    segmenter = cseg.LRAffinityWatershed(threshold_cc, threshold_dt, sigma_seeds,
                                         is_anisotropic=True, invert_input=invert_input)

    # we get the job id from the file name
    overlap_ids = [single_block_watershed(block_id, blocking,
                                          ds_affs, ds_out,
                                          tmp_folder, segmenter) for block_id in block_list]
    overlap_ids = [ids for res in overlap_ids for ids in res]
    job_id = int(os.path.split(block_file)[1].split('_')[2][:-4])
    np.save(os.path.join(tmp_folder, '1_output_ovlps_%i.npy' % job_id), overlap_ids)

    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("aff_path", type=str)
    parser.add_argument("aff_key", type=str)

    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--block_file", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)

    args = parser.parse_args()
    # TODO add the optional arguments
    watershed_step1(args.aff_path, args.aff_key,
                    args.out_path, args.out_key,
                    list(args.block_shape),
                    args.tmp_folder, args.block_file)
