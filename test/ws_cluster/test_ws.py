import sys
sys.path.append('../..')
from cluster_tools.watershed import make_batch_jobs

AFF_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/n5/cremi_warped_sampleA+_predictions.n5'
KEY_XY = 'affs_xy'
KEY_Z = 'averaged_0_1_2_3_4_5_6_7_8_9'
OUT_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/n5/cremi_warped_sampleA+_watersheds.n5'
OUT_KEY = 'seed_averaged_affinities'
TMP_FOLDER = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/tmp_files'
BLOCK_SHAPE = (25, 256, 256)
CHUNKS = (25, 256, 256)
EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


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
