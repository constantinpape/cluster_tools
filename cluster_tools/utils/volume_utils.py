import os
import json
from itertools import product

import numpy as np
import h5py
import z5py
import vigra

from scipy.ndimage.morphology import binary_erosion
from nifty.tools import blocking
from .knossos_wrapper import KnossosFile
from .volume_classes import InterpolatedVolume, TransformedVolume

# use vigra filters as fallback if we don't have
# fastfilters available
try:
    import fastfilters as ff
except ImportError:
    import vigra.filters as ff


def is_z5(path):
    ext = os.path.splitext(path)[1][1:].lower()
    return ext in ('n5', 'zr', 'zarr')


def is_h5(path):
    ext = os.path.splitext(path)[1][1:].lower()
    return ext in ('h5', 'hdf5', 'hdf', 'ilp')


def file_reader(path, mode='a'):
    if is_z5(path):
        return z5py.File(path, mode=mode)
    elif is_h5(path):
        return h5py.File(path, mode=mode)
    else:
        try:
            return KnossosFile(path)
        except RuntimeError:
            ext = os.path.splitext(path)[1][1:].lower()
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
def normalize(input_, min_val=None, max_val=None):
    input_ = input_.astype('float32')
    min_val = input_.min() if min_val is None else min_val
    input_ -= min_val
    max_val = input_.max() if max_val is None else max_val
    if max_val > 0:
        input_ /= max_val
    return input_


def watershed(input_, seeds, size_filter=0, exclude=None):
    ws, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds)
    if size_filter > 0:
        ws, max_id = apply_size_filter(ws, input_, size_filter,
                                       exclude=exclude)
    return ws, max_id


def apply_size_filter(segmentation, input_, size_filter, exclude=None):
    ids, sizes = np.unique(segmentation, return_counts=True)
    filter_ids = ids[sizes < size_filter]
    if exclude is not None:
        filter_ids = filter_ids[np.logical_not(np.in1d(filter_ids, exclude))]
    filter_mask = np.in1d(segmentation, filter_ids).reshape(segmentation.shape)
    segmentation[filter_mask] = 0
    _, max_id = vigra.analysis.watershedsNew(input_, seeds=segmentation, out=segmentation)
    return segmentation, max_id


def _make_checkerboard(blocking):
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


def _make_checkerboard_with_roi(blocking, roi_begin, roi_end):

    # find the smallest roi coordinate
    block0 = blocking.coordinatesToBlockId(roi_begin)

    blocks_a = [block0]
    blocks_b = []
    all_blocks = [block0]

    blocks_in_roi = blocking.getBlockIdsOverlappingBoundingBox(roi_begin, roi_end)
    assert block0 in blocks_in_roi

    def recurse(current_block, insert_list):
        other_list = blocks_a if insert_list is blocks_b else blocks_b
        for dim in range(3):
            ngb_id = blocking.getNeighborId(current_block, dim, False)
            if ngb_id != -1:
                #  check if this block is overlapping the roi
                if ngb_id not in blocks_in_roi:
                    continue
                if ngb_id not in all_blocks:
                    insert_list.append(ngb_id)
                    all_blocks.append(ngb_id)
                    recurse(ngb_id, other_list)

    recurse(block0, blocks_b)
    all_blocks = blocks_a + blocks_b
    expected = set(blocks_in_roi)
    assert len(all_blocks) == len(expected), "%i, %i" % (len(all_blocks), len(expected))
    assert len(set(all_blocks) - expected) == 0
    assert len(blocks_a) == len(blocks_b), "%i, %i" % (len(blocks_a), len(blocks_b))
    return blocks_a, blocks_b


def make_checkerboard_block_lists(blocking, roi_begin=None, roi_end=None):
    assert (roi_begin is None) == (roi_end is None)
    if roi_begin is None:
        return _make_checkerboard(blocking)
    else:
        return _make_checkerboard_with_roi(blocking, roi_begin, roi_end)


def load_mask(mask_path, mask_key, shape):
    with file_reader(mask_path, 'r') as f_mask:
        mshape = f_mask[mask_key].shape
    # check if th mask is at full - shape, otherwise interpolate
    if tuple(mshape) == tuple(shape):
        mask = file_reader(mask_path, 'r')[mask_key]
    else:
        with file_reader(mask_path, 'r') as f_mask:
            mask = f_mask[mask_key][:].astype('bool')
        mask = InterpolatedVolume(mask, shape, spline_order=0)
    return mask


def get_face(blocking, block_id, ngb_id, axis, halo=[1, 1, 1]):
    # get the two block coordinates
    block_a = blocking.getBlock(block_id)
    block_b = blocking.getBlock(ngb_id)
    ndim = len(block_a.shape)

    # validate exhaustively, because a lot of errors may happen here ...
    # validate the non-overlapping axes
    assert all(beg_a == beg_b for dim, beg_a, beg_b
               in zip(range(ndim), block_a.begin, block_b.begin) if dim != axis),\
        "begin_a: %s, begin_b: %s, axis: %i" % (str(block_a.begin), str(block_b.begin), axis)
    assert all(end_a == end_b for dim, end_a, end_b
               in zip(range(ndim), block_a.end, block_b.end) if dim != axis)
    # validate the overlapping axis
    assert block_a.begin[axis] != block_b.begin[axis]
    assert block_a.end[axis] != block_b.end[axis]

    # compute the bounding box corresponiding to the face between the two blocks
    face = tuple(slice(beg, end) if dim != axis else slice(end - ha, end + ha)
                 for dim, beg, end, ha in zip(range(ndim), block_a.begin, block_a.end, halo))
    # get the local coordinates of faces in a and b
    slice_a = slice(0, halo[axis])
    slice_b = slice(halo[axis], 2 * halo[axis])

    face_a = tuple(slice(None) if dim != axis else slice_a
                   for dim in range(ndim))
    face_b = tuple(slice(None) if dim != axis else slice_b
                   for dim in range(ndim))
    return face, face_a, face_b


def iterate_faces(blocking, block_id, halo=[1, 1, 1], return_only_lower=True,
                  empty_blocks=None):

    ndim = len(blocking.blockShape)
    assert len(halo) == ndim, str(halo)
    directions = (False,) if return_only_lower else (False, True)
    # iterate over the axes and directions
    for axis in range(ndim):
        for direction in directions:
            # get neighbor id and check if it is valid
            ngb_id = blocking.getNeighborId(block_id, axis, direction)
            if ngb_id == -1:
                continue
            if empty_blocks is not None:
                if ngb_id in empty_blocks:
                    continue
            face, face_a, face_b = get_face(blocking, block_id, ngb_id,
                                            axis, halo)
            yield face, face_a, face_b, block_id, ngb_id


def faces_to_ovlp_axis(face_a, face_b):
    axis = np.where([fa != fb for fa, fb in zip(face_a, face_b)])[0]
    assert len(axis) == 1, str(axis)
    return axis[0]


def mask_corners(input_, halo):
    ndim = input_.ndim
    shape = input_.shape

    corners = ndim * [[0, 1]]
    corners = product(*corners)

    for corner in corners:
        corner_bb = tuple(slice(0, ha) if co == 0 else slice(sh - ha, sh)
                          for ha, co, sh in zip(halo, shape, corner))
        input_[corner_bb] = 0

    return input_


def preserving_erosion(mask, erode_by):
    eroded = binary_erosion(mask, iterations=erode_by)
    n_foreground = eroded.sum()
    while n_foreground == 0:
        if erode_by == 1:
            return mask
        erode_by //= 2
        eroded = binary_erosion(mask, iterations=erode_by)
        n_foreground = eroded.sum()
    return eroded


def fit_seeds(objs, obj_ids, bg_id, erode_by, max_erode):
    background = objs == 0
    seeds = bg_id * binary_erosion(background, iterations=max_erode)
    seeds = seeds.astype('uint32')
    # insert seeds for the objects
    for obj_id in obj_ids:
        obj_mask = objs == obj_id
        if obj_mask.sum() == 0:
            continue
        # erode theobject mask for seeds, but preserve small seeds
        erode_obj = erode_by if isinstance(erode_by, int) else erode_by[obj_id]
        obj_seeds = preserving_erosion(obj_mask, erode_obj)
        seeds[obj_seeds] = obj_id
    return seeds


def fit_to_hmap_2d(objs, hmap, erode_by, max_erode, obj_ids, bg_id):

    # make the seeds by binary erosion of background and foreground
    seeds = np.zeros_like(hmap, dtype='uint32')
    for z in range(seeds.shape[0]):
        seeds[z] = fit_seeds(objs[z], obj_ids, bg_id, erode_by, max_erode)

    # apply dt before watershed
    hmap = normalize(hmap)
    threshold = .3
    threshd = (hmap > threshold).astype('uint32')

    # 2d
    dt = np.zeros_like(hmap, dtype='float32')
    for z in range(dt.shape[0]):
        dt[z] = vigra.filters.distanceTransform(threshd[z])

    # normalize distances and add up with hmap
    dt = 1. - normalize(dt)
    alpha = .8
    hmap = alpha * hmap + (1. - alpha) * dt

    # 2d
    objs_new = np.zeros_like(objs, dtype='uint32')
    for z in range(objs_new.shape[0]):
        objs_new[z] = vigra.analysis.watershedsNew(hmap[z], seeds=seeds[z])[0]

    return objs_new, obj_ids


def fit_to_hmap_3d(objs, hmap, erode_by, max_erode, obj_ids, bg_id):
    seeds = fit_seeds(objs, obj_ids, bg_id, erode_by, max_erode)

    # apply dt before watershed
    hmap = normalize(hmap)
    threshold = .3
    threshd = (hmap > threshold).astype('uint32')
    dt = vigra.filters.distanceTransform(threshd)

    # normalize distances and add up with hmap
    dt = 1. - normalize(dt)
    alpha = .8
    hmap = alpha * hmap + (1. - alpha) * dt

    objs_new = vigra.analysis.watershedsNew(hmap, seeds=seeds)[0]
    return objs_new, obj_ids


def fit_to_hmap(objs, hmap, erode_by, fit_3d=True):
    # get object ids (excluding 0) and the new background id
    obj_ids = np.unique(objs)
    if 0 in obj_ids:
        obj_ids = obj_ids[1:]
    bg_id = obj_ids[-1] + 1

    if isinstance(erode_by, int):
        max_erode = erode_by
    else:
        max_erode = max(erode_by.values())
        # json always casts keys to str, so we reverse this here
        erode_by = {int(k): v for k, v in erode_by.items()}

    if fit_3d:
        objs_new, obj_ids = fit_to_hmap_3d(objs, hmap, erode_by, max_erode, obj_ids, bg_id)
    else:
        objs_new, obj_ids = fit_to_hmap_2d(objs, hmap, erode_by, max_erode, obj_ids, bg_id)

    # set background to 0
    objs_new[objs_new == bg_id] = 0
    return objs_new, obj_ids
