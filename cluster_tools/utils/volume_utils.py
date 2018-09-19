import json
import numpy as np
import h5py
import z5py
import vigra
import fastfilters
from nifty.tools import blocking


def file_reader(path, mode='a'):
    ending = path.split('.')[-1].lower()
    if ending in ('n5', 'zr', 'zarr'):
        return z5py.File(path, mode=mode)
    elif ending in ('h5', 'hdf5'):
        return h5py.File(path, mode=mode)
    else:
        raise RuntimeError("Invalid file format %s" % ending)


def get_shape(path, key):
    with file_reader(path, 'r') as f:
        shape = f[key].shape
    return shape


def blocks_in_volume(shape, block_shape,
                     roi_begin=None, roi_end=None):
    assert len(shape) == len(block_shape), '%i; %i' % (len(shape), len(block_shape))
    assert (roi_begin is None) == (roi_end is None)
    blocking_ = blocking([0] * len(shape), list(shape), list(block_shape))
    if roi_begin is None:
        return list(range(blocking_.numberOfBlocks))
    else:
        assert roi_end is not None
        roi_end = [sh if re is None else re for re, sh in zip(roi_end, shape)]
        block_list = blocking_.getBlockIdsOverlappingBoundingBox(list(roi_begin),
                                                                 list(roi_end),
                                                                 [0, 0, 0])
        block_list = [bl.tolist() for bl in block_list]
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
        filt = getattr(fastfilters, filter_name)
        return np.concatenate([filt(in_z, sigma)[None] for in_z in input_], axis=0)
    # apply 3d fillter
    else:
        filt = getattr(fastfilters, filter_name)
        return filt(input_, sigma)


# TODO enable channel-wise normalisation
def normalize(input_):
    input_ = input_.astype('float32')
    input_ -= input_.min()
    input_ /= input_.max()
    return input_


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
