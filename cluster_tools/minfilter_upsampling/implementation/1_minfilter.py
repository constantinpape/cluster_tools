#! /usr/bin/python

import os
import time
import argparse
import numpy as np

import z5py
import nifty
import vigra
from scipy.ndimage.filters import minimum_filter


def minfilter_block(block_id, blocking, halo,
                    mask_ds, ds_mask_ds, out_ds,
                    sampling_factor, filter_shape):
    block = blocking.getBlockWithHalo(block_id, halo)
    outer_roi = tuple(slice(beg, end)
                      for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
    inner_roi = tuple(slice(beg, end)
                      for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
    local_roi = tuple(slice(beg, end)
                      for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
    mask1 = mask_ds[outer_roi]

    # get the downsampled mask and upsample it on the fly
    # TODO check the exactly downsampling
    ds_roi = tuple(slice(beg // sa, end // sa)
                   for beg, end, sa in zip(block.outerBlock.begin, block.outerBlock.end, sampling_factor))
    ds_mask = ds_mask_ds[ds_roi]
    mask_resized = vigra.sampling.resize(ds_mask, shape=mask1.shape)

    mask = np.logical_and(mask1, mask_resized)

    min_filter_mask = minimum_filter(mask, size=filter_shape)
    out_ds[inner_roi] = min_filter_mask[local_roi]


def minfilter_step1(mask_path, mask_key,
                    ds_mask_path, ds_mask_key,
                    out_path, out_key,
                    sampling_factor,
                    filter_shape,
                    block_shape,
                    block_file):
    t0 = time.time()
    mask_ds = z5py.File(mask_path)[mask_key]
    ds_mask_ds = z5py.File(ds_mask_ds)[ds_mask_key]
    out_ds = z5py.File(mask_path)[out_key]

    shape = mask_ds.shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))
    block_list = np.load(block_file)

    # TODO is half of the halo really enough halo ?
    halo = list(fshape // 2 for fshape in filter_shape)
    [minfilter_block(block_id, blocking,
                     halo, mask_ds, ds_mask_ds,
                     out_ds, filter_shape) for block_id in block_list]

    job_id = int(os.path.split(block_file)[1].split('_')[2][:-4])
    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_path", type=str)
    parser.add_argument("mask_key", type=str)
    parser.add_argument("ds_mask_path", type=str)
    parser.add_argument("ds_mask_key", type=str)

    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("--filter_shape", nargs=3, type=int)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--sampling_factor", nargs=3, type=int)
    parser.add_argument("--block_file", type=str)

    args = parser.parse_args()
    # TODO add the optional arguments
    minfilter_step1(args.mask_path, args.mask_key,
                    args.ds_mask_path, args.ds_mask_key,
                    args.out_path, args.out_key,
                    list(args.filter_shape),
                    list(args.block_shape),
                    tuple(args.sampling_factor),
                    args.block_file)
