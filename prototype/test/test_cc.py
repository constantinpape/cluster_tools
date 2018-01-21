import sys
import vigra
import z5py
from cremi_tools.viewer.volumina import view
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


def test_cc():
    block_shape = (10, 500, 500)
    out_chunks = (5, 250, 250)
    tmp_folder = './tmp'
    connected_components_ufd('./binary_volume.n5', 'data',
                             './ccs.n5', 'data',
                             block_shape=block_shape,
                             out_chunks=out_chunks,
                             tmp_folder=tmp_folder,
                             n_threads=1)
                             #n_threads=8)


def view_result():
    binary = z5py.File('./binary_volume.n5')['data'][:]
    labeled = z5py.File('./ccs.n5')['data'][:]
    view([binary, labeled])


if __name__ == '__main__':
    test_cc()
    view_result()
    # binary_sub_volume()
