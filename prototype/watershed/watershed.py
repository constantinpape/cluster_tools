import os
# import time
from concurrent import futures
import numpy as np

import vigra
import z5py
import nifty


# TODO correct thresholds
def watershed(aff_path_xy, key_xy, aff_path_z, key_z, out_path, key_out,
              out_chunks, out_blocks, halo=[5, 50, 50],
              threshold_cc=.1, threshold_dt=.25, sigma_seeds=2.):
    assert os.path.exists(aff_path_xy)
    assert os.path.exists(aff_path_z)
    assert all(block % chunk == 0 for chunk, block in zip(out_chunks, out_blocks))
    ds_xy = z5py.File(aff_path_xy)[key_xy]
    ds_z = z5py.File(aff_path_z)[key_z]

    shape = ds_xy.shape
    assert ds_z.shape == shape

    f_out = z5py.File(out_path, use_zarr_format=False)
    if key_out in f_out:
        ds_out = f_out[key_out]
        assert ds_out.chunks == out_chunks, "%s, %s" % (str(ds_out.chunks), str(out_chunks))
        assert ds_out.shape == shape
    else:
        ds_out = f_out.create_dataset(key_out, shape=shape, chunks=out_chunks, dtype='uint64',
                                      compression='gzip')

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(shape))

    def ws_block(block_id):
        block = blocking.getBlockWithHalo(block_id, halo)
        outer_block, inner_block, local_block = block.outerBlock, block.innerBlock, block.innerLocalBlock
        inner_bb = tuple(slice(b, e) for b, e in zip(inner_block.begin, inner_block.end))
        outer_bb = tuple(slice(b, e) for b, e in zip(outer_block.begin, outer_block.end))
        local_bb = tuple(slice(b, e) for b, e in zip(local_block.begin, local_block.end))
        affs_xy = ds_xy[outer_bb]
        affs_z = ds_z[outer_bb]
        affs_z += affs_xy
        affs_z /= 2.
        # TODO figure out the thresholds
        thresholded = affs_xy < threshold_cc
        seeds = vigra.analysis.labelVolumeWithBackground(thresholded.view('uint8'))
        seed_offset = seeds.max() + 1
        # TODO figure out the thresholds
        if sigma_seeds > 0.:
            affs_xy = vigra.filters.gaussianSmoothing(affs_xy, (sigma_seeds / 10., sigma_seeds, sigma_seeds))
        thresholded_dt = affs_xy < threshold_cc
        dt = vigra.filters.distanceTransform(thresholded_dt.view('uint8'), pixel_pitch=(10., 1., 1.))
        seeds_dt = vigra.analysis.localMaxima3D(dt, allowPlateaus=True, allowAtBorder=True, marker=np.nan)
        seeds_dt = vigra.analysis.labelVolumeWithBackground(np.isnan(seeds_dt).view('uint8'))
        seeds_dt[seeds_dt != 0] += seed_offset
        no_seed_mask = seeds == 0
        seeds[no_seed_mask] = seeds_dt[no_seed_mask]
        ws, _ = vigra.analysis.watershedsNew(1. - affs_z, seeds=seeds)
        ws = ws[local_bb]
        ws, max_id, _ = vigra.analysis.relabelConsecutive(ws)
        ds_out[inner_bb] = ws
        return max_id

    with futures.ThreadPool as tp:
        tasks = [tp.submit(ws_block, block_id) for block_id in blocking.numberOfBlocks]
        [t.result() for t in tasks]


if __name__ == '__main__':
    watershed()
