import h5py
import z5py
from nifty.tools import blocking


def file_reader(path, mode='a'):
    ending = path.split('.').lower()
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


def blocks_in_volume(path, key, block_shape,
                     roi_begin=None, roi_end=None):
    assert (roi_begin is None) == (roi_end is None)
    shape = get_shape(path, key)
    blocking_ = blocking([0, 0, 0], list(shape), list(block_shape))
    if roi_begin is None:
        return list(range(blocking.numberOfBlocks))
    else:
        # TODO intersect blocking with roi and return the
        # block ids
        pass
