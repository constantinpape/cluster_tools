import sys
import z5py
import h5py

sys.path.append('..')
from prototype.minfilter import minfilter


def make_ds_mask():
    path_in = '/home/papec/mnt/papec/Work/neurodata_hdd/sampleE/neuropil_mask_s6.h5'
    key_in = 'mask'

    path_out = '/home/papec/mnt/papec/Work/neurodata_hdd/sampleE/neuropil_mask_s6.n5'
    key_out = 'mask'

    with h5py.File(path_in, 'r') as f:
        mask = f[key_in][:]

    f = z5py.File(path_out, use_zarr_format=False)
    ds = f.create_dataset(key_out, dtype='uint8', shape=mask.shape, chunks=(32, 32, 32), compression='gzip')
    ds[:] = mask


def test_minfilter():
    mask_path = '/home/papec/mnt/saalfeldlab/sampleE'
    mask_key = 'masks/initial_mask'
    ds_mask_path = '/home/papec/mnt/papec/Work/neurodata_hdd/sampleE/neuropil_mask_s6.n5'
    ds_mask_key = 'mask'

    raw_path = '/home/papec/mnt/nrs/sample_E/sample_E.n5'
    raw_key = 'volumes/raw/s0'

    sampling_factor = (6, 64, 64)
    filter_shape = (10, 100, 100)
    block_shape = (50, 512, 512)

    minfilter(mask_path, mask_key,
              raw_path, raw_key,
              ds_mask_path, ds_mask_key,
              sampling_factor,
              filter_shape,
              block_shape)


if __name__ == '__main__':
    # make_ds_mask()
    test_minfilter()
