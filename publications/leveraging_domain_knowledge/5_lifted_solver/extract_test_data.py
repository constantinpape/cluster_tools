import z5py
from cremi_tools.viewer.volumina import view


def extract_middle_cutout():
    path = '/g/kreshuk/data/FIB25/data.n5'
    f = z5py.File(path)

    ds = f['volumes/raw/s0']
    ds.n_threads = 8

    shape = ds.shape
    halo = [50, 512, 512]
    shift = [0, -512, -512]
    bb = tuple(slice(sh // 2 - ha - si,
                     sh // 2 + ha - si) for sh, ha, si in zip(shape, halo, shift))
    raw = ds[bb]
    chunks = ds.chunks

    ds = f['volumes/affinities']
    ds.n_threads = 8
    affs = ds[(slice(None),) + bb]

    if False:
        view([raw, affs.transpose((1, 2, 3, 0))])

    out_path = '/g/kreshuk/data/FIB25/cutout.n5'
    f = z5py.File(out_path)
    f.create_dataset('volumes/raw', data=raw, compression='gzip',
                     chunks=chunks, n_threads=8)
    f.create_dataset('volumes/affinities', data=affs, compression='gzip',
                     chunks=(1,) + chunks, n_threads=8)


if __name__ == '__main__':
    extract_middle_cutout()
