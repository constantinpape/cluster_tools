import sys
import vigra
import z5py
from cremi_tools.viewer.volumina import view
from cremi_tools.metrics import voi

sys.path.append('..')
from prototype import connected_components_ufd


def make_binary_volume():
    path = '/home/papec/Work/neurodata_hdd/cremi/sample_A_20160501.hdf'
    clefts = vigra.readHDF5(path, 'volumes/labels/clefts')
    ignore_label = clefts[0, 0, 0]
    clefts[clefts == ignore_label] = 0
    clefts[clefts != 0] = 1
    f = z5py.File('./binary_volume.n5', use_zarr_format=False)
    ds = f.create_dataset('data', dtype='uint8', compression='gzip', shape=clefts.shape, chunks=(5, 250, 250))
    ds[:] = clefts.astype('uint8')


def binary_sub_volume():
    fin = z5py.File('./binary_volume.n5')
    fout = z5py.File('./subvolume.n5', use_zarr_format=False)
    data = fin['data'][:20, :1000, :1000]
    ds = fout.create_dataset('data', shape=data.shape, compression='gzip', chunks=(5, 250, 250), dtype='uint8')
    ds[:] = data


def simple_cc(path):
    vol = z5py.File(path, use_zarr_format=False)['data'][:]
    cc = vigra.analysis.labelVolumeWithBackground(vol)
    return cc


def test_cc(path):
    block_shape = (10, 500, 500)
    out_chunks = (5, 250, 250)
    tmp_folder = './tmp'
    print("Computing ufd connected components:")
    connected_components_ufd(path, 'data',
                             './ccs.n5', 'data',
                             block_shape=block_shape,
                             out_chunks=out_chunks,
                             tmp_folder=tmp_folder,
                             n_threads=8)
    labeled = z5py.File('./ccs.n5')['data'][:]

    print("Computing vigra connected components:")
    labeled_vi = simple_cc(path)
    view([labeled, labeled_vi])

    # FIXME cremi-tools metrics are not working
    # vi = voi(labeled, labeled_vi)
    # print("Have vis:")
    # print(vi)


def view_result(path):
    binary = z5py.File(path)['data'][:]
    labeled = z5py.File('./ccs.n5')['data'][:]
    view([binary, labeled])


if __name__ == '__main__':
    # make_binary_volume()
    # binary_sub_volume()
    path = './binary_volume.n5'
    test_cc(path)
    # view_result(path)
