import os
import argparse
import json

import numpy as np
import z5py
import vigra
import nifty


def process_block(ds, ds_out, block_id, blocking, block_config):
    boundary_threshold = block_config['boundary_threshold']
    distance_threshold = block_config['distance_threshold']
    sigma = block_config['sigma']
    resolution = block_config['resolution']
    aff_slices = block_config['aff_slices']
    aff_slices = [(slice(sl[0], sl[1]),) for sl in aff_slices]
    invert_channels = block_config['invert_channels']
    assert len(aff_slices) == len(invert_channels)

    # TODO double check this
    # compute the correct halo
    # the factor of 2 should not be necessary
    halo = [2 * int(distance_threshold / res)
            for res in resolution]

    block = blocking.getBlockWithHalo(block_id, halo)
    bb = tuple(slice(beg, end) for beg, end
               in zip(block.outerBlock.begin, block.outerBlock.end))

    affs = []
    for aff_slice, inv_channel in zip(aff_slices, invert_channels):
        aff = ds[aff_slice + bb]
        if aff.dtype == np.dtype('uint8'):
            affs = affs.astype('float32')
        if inv_channel:
            aff = 1. - aff
        if aff.ndim == 3:
            aff = aff[None]
        affs.append(aff)
    affs = np.concatenate(affs, axis=0)

    bmap = np.max(affs, axis=0)
    if sigma > 0:
        aniso = resolution[0] / resolution[1]
        sigma_ = (sigma / aniso, sigma, sigma)
        bmap = vigra.filters.gaussianSmoothing(bmap, sigma_)
    bmap = (bmap > boundary_threshold).astype('uint32')

    dt = vigra.filters.distanceTransform(bmap, pixel_pitch=resolution)
    local_bb = tuple(slice(beg, end) for beg, end
                     in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
    dt = dt[local_bb]
    components = vigra.analysis.labelVolumeWithBackground((dt > distance_threshold).view('uint8'))

    out_bb = tuple(slice(beg, end) for beg, end
                   in zip(block.innerBlock.begin, block.innerBlock.end))
    ds_out[out_bb] = components.astype('uint64')

    return int(components.max())


def step1_dt(path, key, out_path, out_key, cache_folder, job_id):
    input_file = os.path.join(cache_folder, '1_config_%i.json' % job_id)
    with open(input_file) as f:
        input_config = json.load(f)
    block_config = input_config['block_config']
    block_shape = input_config['block_shape']

    ds = z5py.File(path)[key]
    ds_out = z5py.File(out_path)[out_key]

    shape = ds.shape
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    max_ids = [process_block(ds, ds_out, block_id, blocking, config)
               for block_id, config in block_config.items()]
    results = {block_id: max_id
               for block_id, max_id in zip(block_config.keys(), max_ids)}

    with open('1_results.json', 'w') as f:
        json.dump(results, f)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
