# TODO make mask from downsampled for sample E similar
# to block list, but upscaling to full resolution
# do this for neuropil mask
import numpy as np
import z5py
from concurrent import futures


def mask_from_downsampled(mask_path, mask_key,
                          path, out_key,
                          raw_key, sampling_factor):
    mask = z5py.File(mask_path)[mask_key][:]
    ds_shape = mask.shape

    shape = z5py.File(path)[raw_key].shape

    shape_to_full = tuple(dss * sf for dss, sf in zip(ds_shape, sampling_factor))
    assert shape == shape_to_full

    f_out = z5py.File(path, use_zarr_format=False)
    if out_key not in f_out:
        ds_out = f_out.create_dataset(out_key, shape=shape, chunks=(25, 256, 256), compression='gzip')
    else:
        ds_out = f_out[out_key]

    def write_mask(z, y, x):
        if mask[z, y, x] == 1:
            roi = np.s_[z * sampling_factor[0]: min((z + 1) * sampling_factor[0], shape[0]),
                        y * sampling_factor[1]: min((y + 1) * sampling_factor[1], shape[1]),
                        x * sampling_factor[2]: min((x + 1) * sampling_factor[2], shape[2])]
            ds_out[roi] = 1

    with futures.ThreadPoolExecutor(20) as tp:
        tasks = [tp.submit(write_mask, z, y, x)
                 for z in range(ds_shape[0]) for y in range(ds_shape[1]) for x in range(ds_shape[2])]
        [t.result() for t in tasks]


if __name__ == '__main__':
    mask_path = ''
    mask_key = ''
    path = ''
    out_key = 'masks/neuropil_mask'
    raw_key = 'gray'
    sampling_factor = ()
    mask_from_downsampled(mask_path, mask_key,
                          path, out_key,
                          raw_key, sampling_factor)
