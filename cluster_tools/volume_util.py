import h5py
import z5py
import nifty


# FIXME this doesn't make much sense, need to abstract this differently
class BaseVolumeTask(object):
    """
    Base class for a volume i/o based task

    Supports chunked volumes in hdf5, n5 and zarr format
    """
    FILE_ENDINGS = {'n5': z5py, 'zr': z5py, 'zarr': z5py,
                    'h5': h5py, 'hdf5': h5py}

    def __init__(self, path, path_in_file):
        ending = path.split('.')[-1]
        assert ending in self.FILE_ENDINGS, ending
        self.io = self.FILE_ENDINGS[ending]
        assert path_in_file

    def get_shape(self):
        pass
