import numpy as np
import vigra
from concurrent import futures

import z5py

completely_black = {'A': [6, 167],
                    'B': [24],
                    'C': [51, 111],
                    'A+': [88],
                    'B+': [],
                    'C+': [51, 111]}


def find_completely_black_slices(sample):
    out_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    raw_ds = z5py.File(out_path, use_zarr_format=False)['raw']
    shape = raw_ds.shape
    chunks = raw_ds.chunks

    def find_completely_black_chunked(z0, z1):
        print("Find completely black for z-range", z0, z1)
        bb = np.s_[z0:z1]
        raw = raw_ds[bb]
        zero_slices = []
        for z in range(raw.shape[0]):
            raw_z = raw[z]
            if np.allclose(raw_z, 0):
                zero_slices.append(z + z0)

        return zero_slices

    z_points = range(0, shape[0] + 1, chunks[0])
    # zero_slices = [tp.submit(z_points[i], z_points[i + 1]) for i in range(len(z_points) - 1)]
    with futures.ThreadPoolExecutor(10) as tp:
        tasks = [tp.submit(find_completely_black_chunked, z_points[i], z_points[i + 1]) for i in range(len(z_points) - 1)]
        zero_slices = [t.result() for t in tasks]

    all_zero_slices = []
    for zsl in zero_slices:
        all_zero_slices.extend(zsl)

    zero_slices = np.unique(all_zero_slices)
    print("Sample", sample, "has zero slices", zero_slices)


def make_mask(sample):
    path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    black_slice_list = completely_black[sample]
    print("Loading raw...")
    f = z5py.File(path, use_zarr_format=False)
    raw = f['raw'][:]
    print("... done")

    print("Generating mask...")
    zero_mask = (raw == 0).astype('uint8')
    print("... done")
    print("Generating components...")
    zero_components = vigra.analysis.labelVolumeWithBackground(zero_mask)
    print("... done")
    print("Counting components...")
    components, counts = np.unique(zero_components, return_counts=True)
    print("... done")

    biggest_components = np.argsort(counts)[::-1]
    sorted_components = components[biggest_components]

    print("Making final mask...")
    n_comps = 0
    mask = np.zeros_like(raw, dtype='uint8')
    for sort_id in sorted_components:
        if n_comps > 0:
            break
        if sort_id == 0:
            continue
        n_comps += 1
        mask[zero_components == sort_id] = 1
    print("... done")
    for black_id in black_slice_list:
        mask[black_id] = mask[black_id - 1]

    print("Writing to n5...")
    ds = f.create_dataset('masks/original_mask',
                          dtype='uint8',
                          compression='gzip',
                          chunks=(25, 256, 256),
                          shape=mask.shape)
    ds[:] = 1 - mask
    print("... done")


def invert_mask(sample):
    path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    f = z5py.File(path, use_zarr_format=False)
    ds = f['masks/original_mask']
    mask = ds[:]
    ds[:] = 1 - mask


if __name__ == '__main__':
    for sample in ('A', 'B', 'C'):
        invert_mask(sample)
    # make_mask('C')
    # find_completely_black_slices('A+')
