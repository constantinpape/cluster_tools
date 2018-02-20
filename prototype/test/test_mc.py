import sys
import z5py
from cremi_tools.viewer.volumina import view

sys.path.append('..')
from prototype import multicut


def test_mc(n_scales=2):
    # path = '/home/papec/Work/neurodata_hdd/testdata.n5'
    path = './testdata.n5'
    multicut(path, 'watershed', './graph.n5', 'graph',
             './features_aff.n5', path, 'multicut',
             [25, 256, 256], n_scales)


def view_mc():
    path = '/home/papec/Work/neurodata_hdd/testdata.n5'
    f = z5py.File(path)
    raw = f['raw'][:]
    ws = f['watershed'][:]
    # affs = f['affs_z'][:]
    mc = f['multicut'][:]

    # view([raw, affs, ws, mc])
    view([raw, ws, mc])


if __name__ == '__main__':
    test_mc(2)
    view_mc()
