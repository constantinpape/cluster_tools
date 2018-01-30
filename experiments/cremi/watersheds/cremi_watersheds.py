import sys
import os
sys.path.append('../../..')
from cluster_tools.masked_watershed import make_batch_jobs


TMP_FOLDER = '/nrs/saalfeld/papec/cache/tmp_cremi_ws'
BLOCK_SHAPE = (50, 512, 512)
CHUNKS = (25, 256, 256)
EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def jobs_for_cluster_test(sample, n_jobs, key_z='affs_z', key_out='ws_seeded_z'):
    path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    assert os.path.exists(path)
    key_xy = 'predictions/affs_xy'
    key_z = 'predictions/%s' % key_z
    mask_key = 'masks/min_filter_mask'
    key_out = 'watersheds/%s' % key_out

    make_batch_jobs(path, key_xy,
                    path, key_z,
                    path, mask_key,
                    path, key_out,
                    TMP_FOLDER, BLOCK_SHAPE,
                    CHUNKS, n_jobs, EXECUTABLE,
                    use_bsub=True,
                    n_threads_ufd=4,
                    eta=[15, 15, 30, 15])


if __name__ == '__main__':
    n_jobs = 80
    jobs_for_cluster_test('A+', n_jobs)
