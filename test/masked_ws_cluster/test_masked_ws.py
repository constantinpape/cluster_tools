import sys
sys.path.append('../..')
from cluster_tools.masked_watershed import make_batch_jobs

PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/testdata.n5'
AFF_KEY = 'full_affs'
MASK_KEY = 'mask'

OUT_KEY = 'watershed'
TMP_FOLDER = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/tmp_files_ws_masked'
BLOCK_SHAPE = (25, 256, 256)
CHUNKS = (25, 256, 256)
EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def jobs_for_cluster_test(n_jobs):
    make_batch_jobs(PATH, AFF_KEY,
                    PATH, MASK_KEY,
                    PATH, OUT_KEY,
                    TMP_FOLDER, BLOCK_SHAPE,
                    CHUNKS, n_jobs, EXECUTABLE,
                    use_bsub=True,
                    n_threads_ufd=2,
                    eta=[5, 5, 5, 5])


if __name__ == '__main__':
    n_jobs = 32
    jobs_for_cluster_test(n_jobs)
