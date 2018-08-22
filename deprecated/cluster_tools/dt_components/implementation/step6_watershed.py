#! /usr/bin/python

import os
import argparse
import json

import numpy as np
import z5py
import vigra
import nifty


def compute_max_seeds(hmap, boundary_threshold,
                      sigma, offset):

    # we only compute the seeds on the smaller crop of the volume
    seeds = np.zeros_like(hmap, dtype='uint32')

    for z in range(seeds.shape[0]):
        # compute distance transform on the full 2d plane
        dtz = vigra.filters.distanceTransform((hmap[z] > boundary_threshold).astype('uint32'))
        if sigma > 0:
            vigra.filters.gaussianSmoothing(dtz, sigma, out=dtz)
        # compute local maxima of the distance transform, then crop
        seeds_z = vigra.analysis.localMaxima(dtz, allowPlateaus=True, allowAtBorder=True, marker=np.nan)
        seeds_z = vigra.analysis.labelImageWithBackground(np.isnan(seeds_z).view('uint8'))
        # add offset to the seeds
        seeds_z[seeds_z != 0] += offset
        offset = seeds_z.max() + 1
        # write seeds to the corresponding slice
        seeds[z] = seeds_z
    return seeds


def run_2d_ws(hmap, seeds, mask, size_filter, extended_seed_offst):
    for z in range(seeds.shape[0]):

        # vigra does not support uint64 seeds
        # for now this should not be an issue,
        # but if we run this on something much bigger (e.g. FAFB)
        # we need to do something clever here
        ws_z = vigra.analysis.watershedsNew(hmap[z], seeds=seeds[z].astype('uint32'))[0]

        # apply size_filter
        if size_filter > 0:
            ids, sizes = np.unique(ws_z, return_counts=True)
            filter_ids = ids[sizes < size_filter]
            # do not filter ids that belong to the extended seeds
            filter_ids = filter_ids[filter_ids > extended_seed_offst]
            filter_mask = np.ma.masked_array(ws_z, np.in1d(ws_z, filter_ids)).mask
            ws_z[filter_mask] = 0
            vigra.analysis.watershedsNew(hmap[z], seeds=ws_z, out=ws_z)

        # set the invalid mask to zero
        ws_z[mask[z]] = 0

        # write the watershed to the seeds
        seeds[z] = ws_z.astype('uint64')
    return seeds, int(seeds.max())


def ws_block(ds_affs, ds_seeds, ds_mask, ds_out,
             blocking, block_id, block_config, offset,
             empty_blocks):

    if block_id in empty_blocks:
        return 0

    boundary_threshold = block_config['boundary_threshold']
    sigma_maxima = block_config['sigma_maxima']
    size_filter = block_config['size_filter']

    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    bb_affs = (slice(1, 3),) + bb

    # load affinities and make heightmap for the watershed
    affs = ds_affs[bb_affs]
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    hmap = np.mean(1. - affs, axis=0)

    # load the extended seeds
    seeds = ds_seeds[bb]

    # load the mask and make the invalid mask by inversion
    mask = ds_mask[bb].astype('bool')
    inv_mask = np.logical_not(mask)
    hmap[inv_mask] = 1

    # get the maxima seeds on 2d distance transform to fill gaps
    # in the extended seeds
    max_seeds = compute_max_seeds(hmap, boundary_threshold, sigma_maxima, offset)

    # add maxima seeds where we don't have seeds from the distance transform components
    # and where we are not in the invalid mask
    unlabeled_in_seeds = np.logical_and(seeds == 0, mask)
    seeds[unlabeled_in_seeds] += max_seeds[unlabeled_in_seeds]

    # run the watershed
    seeds, max_id = run_2d_ws(hmap, seeds, inv_mask, size_filter, offset)

    # find the new seeds and add another offset that will make them unique between blocks
    # (we need to relabel later to make processing efficient !)
    additional_seed_mask = seeds > offset
    block_offset = block_id * np.prod(blocking.blockShape)
    # print("adding offset", block_offset, "to block", block_id, "to mask of size", np.sum(additional_seed_mask))
    seeds[additional_seed_mask] += block_offset

    # write the result
    out_bb = tuple(slice(beg, end) for beg, end
                   in zip(block.begin, block.end))
    ds_out[out_bb] = seeds


def step6_ws(path, aff_key, seed_key, mask_key, out_key,
             cache_folder, job_id):

    f = z5py.File(path)
    ds_affs = f[aff_key]
    ds_seeds = f[seed_key]
    ds_mask = f[mask_key]
    ds_out = f[out_key]

    input_file = os.path.join(cache_folder, '1_config_%i.json' % job_id)
    with open(input_file) as f:
        input_config = json.load(f)
        block_shape = input_config['block_shape']
        block_config = input_config['block_config']

    offsets_path = os.path.join(cache_folder, 'block_offsets.json')
    with open(offsets_path) as f:
        offset_config = json.load(f)
        empty_blocks = offset_config['empty_blocks']
        offset = offset_config['n_labels']

    shape = ds_out.shape
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    [ws_block(ds_affs, ds_seeds, ds_mask, ds_out,
              blocking, int(block_id), config, offset,
              empty_blocks)
     for block_id, config in block_config.items()]

    # results = {block_id: max_id
    #            for block_id, max_id in zip(block_config.keys(), max_ids)}
    # out_path = os.path.join(cache_folder, '6_results_%i.json' % job_id)
    # with open(out_path, 'w') as f:
    #     json.dump(results, f)

    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('aff_key', type=str)
    parser.add_argument('seed_key')
    parser.add_argument('mask_key', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('job_id', type=int)
    args = parser.parse_args()

    step6_ws(args.path, args.aff_key,
             args.seed_key, args.mask_key,
             args.out_key,
             args.cache_folder, args.job_id)
