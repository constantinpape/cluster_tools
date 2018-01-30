import sys
sys.path.append('../..')
from cluster_tools.masked_watershed import make_batch_jobs

AFF_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/affs_masked.n5'
KEY_XY = 'affs_xy'
KEY_Z = 'affs_z'
MASK_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/mask.n5'
MASK_KEY = 'data'

OUT_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/ws_masked.n5'
OUT_KEY = 'data'
TMP_FOLDER = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/tmp_files_ws_masked'
BLOCK_SHAPE = (25, 256, 256)
CHUNKS = (25, 256, 256)
EXECUTABLE = '/home/papec/Work/software/conda/miniconda2/envs/production/bin/python'


def jobs_for_cluster_test(n_jobs):
    make_batch_jobs(AFF_PATH, KEY_XY,
                    AFF_PATH, KEY_Z,
                    MASK_PATH, MASK_KEY,
                    OUT_PATH, OUT_KEY,
                    TMP_FOLDER, BLOCK_SHAPE,
                    CHUNKS, n_jobs, EXECUTABLE,
                    use_bsub=False,
                    n_threads_ufd=4,
                    eta=[15, 15, 30, 15])


if __name__ == '__main__':
    n_jobs = 4
    jobs_for_cluster_test(n_jobs)
