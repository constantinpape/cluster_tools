import sys
sys.path.append('../..')
from cluster_tools.watershed import make_batch_jobs

PATH_XY = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/binary_volume.n5'
KEY_XY = 'data'
PATH_Z = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/binary_volume.n5'
KEY_Z = 'data'
OUT_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/cc.n5'
OUT_KEY = 'data'
TMP_FOLDER = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/tmp_files'
BLOCK_SHAPE = (50, 512, 512)
CHUNKS = (25, 256, 256)
EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def jobs_for_cluster_test(n_jobs):
    make_batch_jobs(PATH_XY, KEY_XY, PATH_Z, KEY_Z, OUT_PATH, OUT_KEY, TMP_FOLDER,
                    BLOCK_SHAPE, CHUNKS, n_jobs, EXECUTABLE, use_bsub=True)


if __name__ == '__main__':
    n_jobs = 12
    jobs_for_cluster_test(n_jobs)
