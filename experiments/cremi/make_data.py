import h5py
from z5py.converter import convert_h5_to_n5


def convert_sample(sample):
    path = '/nrs/saalfeld/saalfelds/cremi/sample_%s_20160501.aligned.hdf' % sample

    out_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample

    print("Converting raw data...")
    in_key = 'volumes/raw'
    out_key = 'raw'
    convert_h5_to_n5(path, out_path, in_key, out_key, compression='gzip', out_chunks=(25, 256, 256), n_threads=20)

    print("Converting labels...")
    in_key = 'volumes/labels/neuron_ids'
    out_key = 'groundtruth'
    with h5py.File(path, 'r') as f:
        have_gt = in_key in f
    if have_gt:
        convert_h5_to_n5(path, out_path, in_key, out_key, compression='gzip', out_chunks=(25, 256, 256), n_threads=20)


if __name__ == '__main__':
    convert_sample('B')
    convert_sample('C')
