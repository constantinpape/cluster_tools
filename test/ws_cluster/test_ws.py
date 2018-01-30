import sys
sys.path.append('../..')
from cluster_tools.watershed import make_batch_jobs

AFF_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/affs.n5'
KEY_XY = 'affs_xy'
KEY_Z = 'affs_z'
OUT_PATH = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/ws.n5'
OUT_KEY = 'data'
TMP_FOLDER = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/tmp_files'
BLOCK_SHAPE = (25, 256, 256)
CHUNKS = (25, 256, 256)
EXECUTABLE = '/home/papec/Work/software/conda/miniconda2/envs/production/bin/python'


def jobs_for_cluster_test(n_jobs):
    make_batch_jobs(AFF_PATH, KEY_XY,
                    AFF_PATH, KEY_Z,
                    OUT_PATH, OUT_KEY,
                    TMP_FOLDER, BLOCK_SHAPE,
                    CHUNKS, n_jobs, EXECUTABLE, use_bsub=True,
                    n_threads_ufd=4,
                    eta=[15, 15, 30, 15])


if __name__ == '__main__':
    n_jobs = 80
    jobs_for_cluster_test(n_jobs)
