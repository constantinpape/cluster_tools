import sys
from cremi_tools.viewer.volumina import view
import z5py
import numpy as np

sys.path.append('..')
from prototype import watershed


def make_testdata():
    raw_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/n5/cremi_warped_sampleA+.n5'
    aff_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/gp_tf_predictions_iter_400000/cremi_warped_sampleA+_predictions.n5'

    bb = np.s_[10:60, 1000:1512, 1000:1512]
    raw = z5py.File(raw_path)['data'][bb]
    f_out = z5py.File('./raw.n5', use_zarr_format=False)
    ds_raw = f_out.create_dataset('data', shape=raw.shape, dtype='uint8', compression='gzip', chunks=(20, 256, 256))
    ds_raw[:] = raw

    aff_out = z5py.File('./affs.n5', use_zarr_format=False)
    aff_xy = z5py.File(aff_path)['affs_xy'][bb]
    ds_xy = aff_out.create_dataset('affs_xy', shape=aff_xy.shape, dtype='float32', compression='gzip', chunks=(20, 256, 256))
    ds_xy[:] = aff_xy

    aff_z = z5py.File(aff_path)['affs_z'][bb]
    ds_z = aff_out.create_dataset('affs_z', shape=aff_xy.shape, dtype='float32', compression='gzip', chunks=(20, 256, 256))
    ds_z[:] = aff_z


def view_result():
    raw = z5py.File('./raw.n5')['data'][:]
    ws = z5py.File('./ws.n5')['data'][:]
    affs = z5py.File('./affs.n5')['affs_xy'][:]
    view([raw, affs, ws])


def test_ws():
    aff_path = './affs.n5'
    key_xy = 'affs_xy'
    key_z = 'affs_z'
    out_path = './ws.n5'
    key = 'data'
    watershed(aff_path, key_xy, aff_path, key_z, out_path, key,
              out_chunks=(10, 128, 128), out_blocks=(20, 256, 256), tmp_folder='./tmp')


if __name__ == '__main__':
    test_ws()
    view_result()
