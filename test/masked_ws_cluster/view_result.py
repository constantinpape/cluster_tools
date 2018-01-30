import sys
import z5py

sys.path.append('/home/papec/Work/my_projects/cremi_tools')
from cremi_tools.viewer.volumina import view

RAW_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/raw_masked.n5'
AFF_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/affs_masked.n5'
MASK_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/mask.n5'
WS_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/ws_masked.n5'


def view_result():
    raw = z5py.File(RAW_PATH)['data'][:]
    ws = z5py.File(WS_PATH)['data'][:]
    affs = z5py.File(AFF_PATH)['affs_xy'][:]
    mask = z5py.File(MASK_PATH)['data'][:]
    view([raw, affs, ws, mask],
         ['raw', 'affs', 'ws', 'mask'])


if __name__ == '__main__':
    view_result()
