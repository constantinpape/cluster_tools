#! /usr/bin/python

import os
import argparse
import json

import numpy as np
import z5py
import vigra
import nifty


def compute_dt_components(affs, mask,
                          boundary_threshold, distance_threshold,
                          sigma, resolution, local_bb):
    # transform affinities to boundary map via max projection
    # and (optional) smoothing
    bmap = np.max(affs, axis=0)
    if sigma > 0:
        aniso = resolution[0] / resolution[1]
        sigma_ = (sigma / aniso, sigma, sigma)
        bmap = vigra.filters.gaussianSmoothing(bmap, sigma_)
    bmap = (bmap > boundary_threshold).astype('uint32')

    # set the inverse mask to 1
    bmap[mask] = 1

    dt = vigra.filters.distanceTransform(bmap, pixel_pitch=resolution)
    dt = dt[local_bb]
    return vigra.analysis.labelVolumeWithBackground((dt > distance_threshold).view('uint8'))


def process_block(ds, ds_out, ds_mask, block_id, blocking, block_config):

    # get the configuration for this block
    boundary_threshold = block_config['boundary_threshold']
    distance_threshold = block_config['distance_threshold']
    sigma_components = block_config['sigma_components']
    resolution = block_config['resolution']
    aff_slices = block_config['aff_slices']
    aff_slices = [(slice(sl[0], sl[1]),) for sl in aff_slices]
    invert_channels = block_config['invert_channels']
    assert len(aff_slices) == len(invert_channels)

    # for debugging
    # if block_id == 0:
    #     for cnf, val in block_config.items():
    #         print(cnf, val)

    # TODO double check this
    # compute the correct halo
    # the factor of 2 should not be necessary
    halo = [2 * int(distance_threshold / res)
            for res in resolution]

    block = blocking.getBlockWithHalo(block_id, halo)
    bb = tuple(slice(beg, end) for beg, end
               in zip(block.outerBlock.begin, block.outerBlock.end))

    # first load the mask and see if we have to do anything
    mask = ds_mask[bb].astype('bool')
    local_bb = tuple(slice(beg, end) for beg, end
                     in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
    inner_mask = mask[local_bb]
    if np.sum(inner_mask) == 0:
        return 0

    # load the affinities from the slices specified in the config
    affs = []
    for aff_slice, inv_channel in zip(aff_slices, invert_channels):
        aff = ds[aff_slice + bb]
        if aff.dtype == np.dtype('uint8'):
            aff = aff.astype('float32') / 255.
        if inv_channel:
            aff = 1. - aff
        if aff.ndim == 3:
            aff = aff[None]
        affs.append(aff)
    affs = np.concatenate(affs, axis=0)

    # compute the components of the thresholded (3d, anisotropic) distance transform
    inv_mask = np.logical_not(mask)
    seeds = compute_dt_components(affs, inv_mask,
                                  boundary_threshold, distance_threshold,
                                  sigma_components, resolution, local_bb)

    out_bb = tuple(slice(beg, end) for beg, end
                   in zip(block.innerBlock.begin, block.innerBlock.end))
    ds_out[out_bb] = seeds.astype('uint64')

    return int(seeds.max()) + 1


def step1_dt(path, aff_key, out_key, mask_key, cache_folder, job_id):
    input_file = os.path.join(cache_folder, '1_config_%i.json' % job_id)
    with open(input_file) as f:
        input_config = json.load(f)
    block_config = input_config['block_config']
    block_shape = input_config['block_shape']

    f = z5py.File(path)
    ds = f[aff_key]
    ds_out = f[out_key]
    ds_mask = f[mask_key]

    shape = ds_out.shape
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    max_ids = [process_block(ds, ds_out, ds_mask, int(block_id), blocking, config)
               for block_id, config in block_config.items()]
    results = {block_id: max_id
               for block_id, max_id in zip(block_config.keys(), max_ids)}

    with open(os.path.join(cache_folder, '1_results_%i.json' % job_id), 'w') as f:
        json.dump(results, f)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('aff_key', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('mask_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('job_id', type=int)
    args = parser.parse_args()

    step1_dt(args.path, args.aff_key, args.out_key,
             args.mask_key, args.cache_folder, args.job_id)
