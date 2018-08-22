from math import floor, ceil

import numpy as np
from scipy.ndimage.filters import minimum_filter

import vigra
import z5py
import nifty

from cremi_tools.viewer.volumina import view


# FIXME check projection to s0 and resampling
def minfilter_block(block_id, blocking, halo,
                    mask_ds, raw_ds, ds_mask_ds,
                    sampling_factor, filter_shape):
    block = blocking.getBlockWithHalo(block_id, halo)
    outer_roi = tuple(slice(beg, end)
                      for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
    mask1 = mask_ds[outer_roi]

    # get the downsampled mask and upsample it on the fly
    # TODO check the exactly downsampling
    begin_ds = tuple(beg / sf for beg, sf in zip(block.outerBlock.begin, sampling_factor))
    end_ds = tuple(beg / sf for beg, sf in zip(block.outerBlock.end, sampling_factor))
    ds_roi = tuple(slice(int(floor(beg)), int(ceil(end))) for beg, end in zip(begin_ds, end_ds))

    ds_mask = ds_mask_ds[ds_roi]

    print(ds_mask.shape)
    print(tuple(dshape * sa for dshape, sa in zip(ds_mask.shape, sampling_factor)))
    print(tuple(mask1.shape))
    print()

    mask_resized = vigra.sampling.resize(ds_mask.astype('float32'), shape=mask1.shape, order=1)
    mask_resized = mask_resized > 0

    mask = np.logical_and(mask1, mask_resized)
    min_filter_mask = minimum_filter(mask, size=filter_shape)

    raw = raw_ds[outer_roi]

    view([raw, min_filter_mask, mask, mask1, mask_resized],
         ['raw', 'min filter mask', 'combined mask', 'block mask', 'resized mask'])


def find_interesting_blocks(blocking, ds_mask_ds, sampling_factor, halo, n_blocks=5):

    block_list = []
    block_ids = range(blocking.numberOfBlocks)
    for block_id in block_ids:
        block = blocking.getBlockWithHalo(block_id, halo)
        ds_roi = tuple(slice(beg // sa, end // sa)
                       for beg, end, sa in zip(block.outerBlock.begin, block.outerBlock.end, sampling_factor))
        ds_mask = ds_mask_ds[ds_roi]
        vals = np.unique(ds_mask)
        if len(vals) == 2:
            block_list.append(block_id)
            if len(block_list) > n_blocks:
                break

    return block_list


def minfilter(mask_path, mask_key,
              raw_path, raw_key,
              ds_mask_path, ds_mask_key,
              sampling_factor,
              filter_shape,
              block_shape):

    raw_ds = z5py.File(raw_path)[raw_key]
    mask_ds = z5py.File(mask_path)[mask_key]
    shape = mask_ds.shape

    ds_mask_ds = z5py.File(ds_mask_path)[ds_mask_key]
    ds_shape = ds_mask_ds.shape
    shape_from_ds = tuple(sh // sf for sh, sf in zip(shape, sampling_factor))

    assert shape_from_ds == ds_shape, "%s, %s" % (str(shape_from_ds), str(ds_shape))

    shape = mask_ds.shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))

    # TODO is half of the halo really enough halo ?
    halo = list(fshape // 2 for fshape in filter_shape)

    print("Finding interesting blocks...")
    block_ids = find_interesting_blocks(blocking, ds_mask_ds, sampling_factor, halo)
    print("...done")

    [minfilter_block(block_id, blocking, halo,
                     mask_ds, raw_ds, ds_mask_ds,
                     sampling_factor, filter_shape) for block_id in block_ids]
