import sys

sys.path.append('..')
from prototype import multicut


def test_mc():
    path = '/home/papec/Work/neurodata_hdd/testdata.n5'
    multicut(path, 'watersheds', './graph.n5', 'graph',
             './features_aff.n5', path, 'multicut',
             [25, 256, 256], 2)


if __name__ == '__main__':
    test_mc()
