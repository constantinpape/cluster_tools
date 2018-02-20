import sys
import os

sys.path.append('..')
from prototype import features

OFFSETS = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
           [-2, 0, 0], [0, -3, 0], [0, 0, -3],
           [-3, 0, 0], [0, -9, 0], [0, 0, -9],
           [-4, 0, 0], [0, -27, 0], [0, 0, -27]]


def test_features(use_affs):
    path = './testdata.n5'
    # path = '/home/papec/Work/neurodata_hdd/testdata.n5'
    assert os.path.exists(path)
    data_key = 'affs_xy' if not use_affs else 'full_affs'
    seg_key = 'watershed'
    features_out = './features_aff.n5' if use_affs else './features_bmap.n5'
    features('./graph.n5', path, data_key,
             path, seg_key, features_out,
             OFFSETS if use_affs else None)


if __name__ == '__main__':
    test_features(True)
