import os
import json
from functools import partial
from math import floor, ceil

import numpy as np
import h5py
import z5py
import vigra

# use vigra filters as fallback if we don't have
# fastfilters available
try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff

from nifty.tools import blocking


def is_z5(path):
    ext = os.path.splitext(path)[1][1:].lower()
    return ext in ('n5', 'zr', 'zarr')


def is_h5(path):
    ext = os.path.splitext(path)[1][1:].lower()
    return ext in ('h5', 'hdf5', 'hdf')


def file_reader(path, mode='a'):
    if is_z5(path):
        return z5py.File(path, mode=mode)
    elif is_h5(path):
        return h5py.File(path, mode=mode)
    else:
        raise RuntimeError("Invalid file format %s" % ext)


def get_shape(path, key):
    with file_reader(path, 'r') as f:
        shape = f[key].shape
    return shape


def blocks_in_volume(shape, block_shape,
                     roi_begin=None, roi_end=None,
                     block_list_path=None):
    assert len(shape) == len(block_shape), '%i; %i' % (len(shape), len(block_shape))
    assert (roi_begin is None) == (roi_end is None)
    have_roi = roi_begin is not None
    have_path = block_list_path is not None
    if have_path:
        assert os.path.exists(block_list_path),\
            "Was given block_list_path %s that doesn't exist" % block_list_path

    blocking_ = blocking([0] * len(shape), list(shape), list(block_shape))

    # we don't have a roi and don't have a block_list_path
    # -> return all block_ids
    if not have_roi and not block_list_path:
        return list(range(blocking_.numberOfBlocks))

    # if we have a roi load the blocks in roi
    if have_roi:
        roi_end = [sh if re is None else re for re, sh in zip(roi_end, shape)]
        block_list = blocking_.getBlockIdsOverlappingBoundingBox(list(roi_begin),
                                                                 list(roi_end))
        block_list = block_list.tolist()
        assert len(block_list) == len(set(block_list)), "%i, %i" % (len(block_list), len(set(block_list)))

    # if we have a block list path, load it
    if have_path:
        with open(block_list_path) as f:
            list_from_path = json.load(f)
        # if we have a roi, need to intersect
        if have_roi:
            block_list = np.intersect1d(list_from_path, block_list).tolist()
        else:
            block_list = list_from_path

    return block_list


def block_to_bb(block):
    return tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))


def apply_filter(input_, filter_name, sigma, apply_in_2d=False):
    # apply 3d filter with anisotropic sigma - only supported in vigra
    if isinstance(sigma, (tuple, list)):
        assert len(sigma) == input_.ndim
        assert not apply_in_2d
        filt = getattr(vigra.filters, filter_name)
        return filt(input_, sigma)
    # apply 2d filter to individual slices
    elif apply_in_2d:
        filt = getattr(ff, filter_name)
        return np.concatenate([filt(in_z, sigma)[None] for in_z in input_], axis=0)
    # apply 3d fillter
    else:
        filt = getattr(ff, filter_name)
        return filt(input_, sigma)


# TODO enable channel-wise normalisation
def normalize(input_):
    input_ = input_.astype('float32')
    input_ -= input_.min()
    input_ /= input_.max()
    return input_


def watershed(input_, seeds, size_filter=0, exclude=None):
    ws, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds)
    if size_filter > 0:
        ws, max_id = apply_size_filter(ws, input_, size_filter,
                                       exclude=exclude)
    return ws.astype('uint64'), max_id


def apply_size_filter(segmentation, input_, size_filter, exclude=None):
    ids, sizes = np.unique(segmentation, return_counts=True)
    filter_ids = ids[sizes < size_filter]
    if exclude is not None:
        filter_ids = filter_ids[np.logical_not(np.in1d(filter_ids, exclude))]
    filter_mask = np.in1d(segmentation, filter_ids).reshape(segmentation.shape)
    segmentation[filter_mask] = 0
    _, max_id = vigra.analysis.watershedsNew(input_, seeds=segmentation, out=segmentation)
    return segmentation, max_id


# TODO is there a more efficient way to do this?
# TODO support roi
def make_checkerboard_block_lists(blocking, roi_begin=None, roi_end=None):
    assert (roi_begin is None) == (roi_end is None)
    if roi_begin is not None:
        raise NotImplementedError("Roi not implemented")
    blocks_a = [0]
    blocks_b = []
    all_blocks = [0]

    def recurse(current_block, insert_list):
        other_list = blocks_a if insert_list is blocks_b else blocks_b
        for dim in range(3):
            ngb_id = blocking.getNeighborId(current_block, dim, False)
            if ngb_id != -1:
                if ngb_id not in all_blocks:
                    insert_list.append(ngb_id)
                    all_blocks.append(ngb_id)
                    recurse(ngb_id, other_list)

    recurse(0, blocks_b)
    all_blocks = blocks_a + blocks_b
    expected = set(range(blocking.numberOfBlocks))
    assert len(all_blocks) == len(expected), "%i, %i" % (len(all_blocks), len(expected))
    assert len(set(all_blocks) - expected) == 0
    assert len(blocks_a) == len(blocks_b), "%i, %i" % (len(blocks_a), len(blocks_b))
    return blocks_a, blocks_b


class InterpolatedVolume(object):
    def __init__(self, volume, output_shape, spline_order=0):
        assert isinstance(volume, np.ndarray)
        assert len(output_shape) == volume.ndim == 3, "Only 3d supported"
        assert all(osh > vsh for osh, vsh in zip(output_shape, volume.shape)),\
            "Can only interpolate to larger shapes, got %s %s" % (str(output_shape), str(volume.shape))
        self.volume = volume
        self.shape = output_shape
        self.dtype = volume.dtype

        self.scale = [sh / float(fsh) for sh, fsh in zip(self.volume.shape, self.shape)]
        if np.dtype(self.dtype) == np.bool:
            self.min, self.max = 0, 1
        else:
            try:
                self.min = np.iinfo(np.dtype(self.dtype)).min
                self.max = np.iinfo(np.dtype(self.dtype)).max
            except ValueError:
                self.min = np.finfo(np.dtype(self.dtype)).min
                self.max = np.finfo(np.dtype(self.dtype)).max

        self.interpol_function = partial(vigra.sampling.resize, order=spline_order)

    def _interpolate(self, data, shape):
        data = self.interpol_function(data.astype('float32'), shape=shape)
        np.clip(data, self.min, self.max, out=data)
        return data.astype(self.dtype)

    def _normalize_index(self, index):
        if isinstance(index, slice):
            index = (index,)
        else:
            assert isinstance(index, tuple)
            assert len(index) <= len(self.shape)
            assert all(isinstance(ind, slice) for ind in index)

        if len(index) < len(self.shape):
            n_missing = len(self.shape) - len(index)
            index = index + n_missing * (slice(None),)
        index = tuple(slice(0 if ind.start is None else ind.start,
                            sh if ind.stop is None else ind.stop)
                      for ind, sh in zip(index, self.shape))
        return index

    def __getitem__(self, index):
        index = self._normalize_index(index)
        ret_shape = tuple(ind.stop - ind.start for ind in index)
        index_ = tuple(slice(int(floor(ind.start * sc)),
                             int(ceil(ind.stop * sc))) for ind, sc in zip(index, self.scale))
        # vigra can't deal with singleton dimension
        small_shape = tuple(idx.stop - idx.start for idx in index_)
        index_ = tuple(slice(idx.start, idx.stop) if sh > 1 else
                       slice(idx.start, idx.stop + 1) for idx, sh in zip(index_, small_shape))
        data = self.volume[index_]

        # speed ups for empty blocks and masks
        dsum = data.sum()
        if dsum == 0:
            return np.zeros(ret_shape, dtype=self.dtype)
        elif dsum == data.size:
            return np.ones(ret_shape, dtype=self.dtype)
        return self._interpolate(data, ret_shape)

    def __setitem__(self, index, item):
        raise NotImplementedError("Setitem not implemented")


def load_mask(mask_path, mask_key, shape):
    with file_reader(mask_path, 'r') as f_mask:
        mshape = f_mask[mask_key].shape

    # check if th mask is at full - shape, otherwise interpolate
    if tuple(mshape) == tuple(shape):
        # TODO this only works for n5
        mask = z5py.File(mask_path)[mask_key]

    else:
        with file_reader(mask_path, 'r') as f_mask:
            mask = f_mask[mask_key][:].astype('bool')
        mask = InterpolatedVolume(mask, shape, spline_order=0)
    return mask
