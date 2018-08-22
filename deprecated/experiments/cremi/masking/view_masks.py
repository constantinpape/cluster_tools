import sys
import os
sys.path.append('/home/papec/Work/my_projects/cremi_tools')
from cremi_tools.viewer.bdv import view


def view_masks(sample):
    path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cremi_warped/sample%s.n5' % sample
    assert os.path.exists(path), path
    raw_key = 'raw'
    aff_key = 'predictions/affs_xy'
    mask_key = 'masks/original_mask'
    out_key = 'masks/min_filter_mask'

    view([path, path, path, path],
         [raw_key, aff_key, mask_key, out_key],
         [(0, 255), (0, 1), (0, 1), (0, 1)],
         resolution=[1, 1, 10])


if __name__ == '__main__':
    sample = 'A+'
    view_masks(sample)
