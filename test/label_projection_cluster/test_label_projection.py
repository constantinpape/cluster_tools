import sys
sys.path.append('../..')
from cluster_tools.label_projection import make_batch_jobs

PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/testdata.n5'

LABELS_KEY = 'watershed'
LABELING_KEY = 'node_labeling_multicut'
OUT_KEY = 'segmentation_multicut'

TMP_FOLDER = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/tmp_files_labelprojection'
BLOCK_SHAPE = (25, 256, 256)
EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def jobs_for_cluster_test(n_jobs):
    make_batch_jobs(PATH, LABELS_KEY,
                    PATH, OUT_KEY,
                    PATH, LABELING_KEY,
                    TMP_FOLDER,
                    BLOCK_SHAPE, BLOCK_SHAPE, n_jobs,
                    executable=EXECUTABLE,
                    use_bsub=False,
                    eta=5)


if __name__ == '__main__':
    n_jobs = 8
    jobs_for_cluster_test(n_jobs)
