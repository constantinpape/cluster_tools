import sys
import os

sys.path.append('..')
from prototype import compute_region_graph


def test_graph():
    # seg_path = './testdata.n5'
    seg_path = '/home/papec/Work/neurodata_hdd/testdata.n5'
    assert os.path.exists(seg_path)
    seg_key = 'watershed'
    blocks = (25, 256, 256)
    compute_region_graph(seg_path, seg_key, blocks, './graph.n5')


if __name__ == '__main__':
    test_graph()
