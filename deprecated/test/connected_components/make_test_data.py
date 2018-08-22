import vigra
import z5py


def make_test_data(in_path, in_key, out_path, out_key):
    data = vigra.readHDF5(in_path, in_key)
    ignore_value = data[0, 0, 0]
    data[data == ignore_value] = 0
    data[data != 0] = 1
    f_out = z5py.File(out_path, use_zarr_format=False)
    ds_out = f_out.create_dataset(out_key, dtype='uint8', compression='gzip', shape=data.shape,
                                  chunks=(25, 256, 256))
    ds_out[:] = data.astype('uint8')


if __name__ == '__main__':
    make_test_data('/home/papec/Work/neurodata_hdd/cremi/sample_A_padded_20160501.hdf',
                   'volumes/labels/neuron_ids',
                   './binary_volume.n5', 'data')
