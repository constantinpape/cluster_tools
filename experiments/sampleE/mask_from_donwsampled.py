# TODO make mask from downsampled for sample E similar
# to block list, but upscaling to full resolution
# do this for neuropil mask
import numpy as np
import z5py
import h5py
from concurrent import futures


def mask_from_downsampled(mask_path, mask_key,
                          path, out_key,
                          raw_key, sampling_factor):
    # mask = z5py.File(mask_path)[mask_key][:]
    with h5py.File(mask_path, 'r') as f:
        mask = f[mask_key][:]
    ds_shape = mask.shape

    shape = z5py.File(path)[raw_key].shape

    # shape_to_full = tuple(dss * sf for dss, sf in zip(ds_shape, sampling_factor))
    # the shapes don't match up perfectly ...
    # assert shape == shape_to_full, "%s, %s" % (str(shape), str(shape_to_full))

    f_out = z5py.File(path, use_zarr_format=False)
    if out_key not in f_out:
        ds_out = f_out.create_dataset(out_key, shape=shape, chunks=(26, 256, 256), compression='gzip', dtype='uint8')
    else:
        ds_out = f_out[out_key]

    def write_mask(z_pair):
        print("Processing z_pair", z_pair)
        for z in z_pair:
            for y in range(ds_shape[1]):
                for x in range(ds_shape[2]):
                    if mask[z, y, x] == 1:
                        roi = np.s_[z * sampling_factor[0]: min((z + 1) * sampling_factor[0], shape[0]),
                                    y * sampling_factor[1]: min((y + 1) * sampling_factor[1], shape[1]),
                                    x * sampling_factor[2]: min((x + 1) * sampling_factor[2], shape[2])]
                        roi_shape = tuple(c.stop - c.start for c in roi)
                        ds_out[roi] = np.ones(roi_shape, dtype='uint8')

    z_range = range(ds_shape[0])
    with futures.ThreadPoolExecutor(80) as tp:
        tasks = [tp.submit(write_mask, (z, z+1)) for z in z_range[::2]]
        [t.result() for t in tasks]


if __name__ == '__main__':
    mask_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/sampleE/neuropil_mask_s7.h5'
    mask_key = 'data'
    path = '/groups/saalfeld/saalfeldlab/sampleE'
    out_key = 'masks/neuropil_mask'
    raw_key = 'masks/initial_mask'
    sampling_factor = (13, 128, 128)
    mask_from_downsampled(mask_path, mask_key,
                          path, out_key,
                          raw_key, sampling_factor)
