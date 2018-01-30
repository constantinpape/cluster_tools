import numpy as np
import vigra

import z5py

completely_black = {'A+': [88],
                    'B+': [],
                    'C+': []}


def make_mask(sample):
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sample%s.n5' % sample
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
    ds = f.create_dataset('original_mask',
                          dtype='uint8',
                          compressor='gzip',
                          chunks=(25, 256, 256),
                          shape=mask.shape)
    ds[:] = mask
    print("... done")


if __name__ == '__main__':
    make_mask('A+')
