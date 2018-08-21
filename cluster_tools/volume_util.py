import json
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
    assert len(shape) == len(block_shape) == 3, '%i; %i' % (len(shape), len(block_shape))
    assert (roi_begin is None) == (roi_end is None)
    blocking_ = blocking([0, 0, 0], list(shape), list(block_shape))
    if roi_begin is None:
        return list(range(blocking_.numberOfBlocks))
    else:
        # TODO intersect blocking with roi and return the
        # block ids
        pass


def block_to_bb(block):
    return tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))


# TODO enable applying 2d filter to 3d input
def apply_filter(input_, filter_name, sigma):
    if isinstance(sigma, (tuple, list)):
        assert len(sigma) == input_.ndim
        filt = getattr(vigra.filters, filter_name)
    else:
        filt = getattr(fastfilters, filter_name)
    return filt(input_, sigma)


# TODO enable channel-wise normalisation
def normalize(input_):
    input_ = input_.astype('float32')
    input_ -= input_.min()
    input_ /= input_.max()
    return input_


def apply_size_filter(segmentation, input_, size_filter):
    pass
