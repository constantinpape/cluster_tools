import os
from shutil import rmtree
import sys
sys.path.append('../..')
from cluster_tools.graph import make_batch_jobs

# LABELS_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/ws.n5'
LABELS_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sampleA+.n5'

# LABELS_KEY = 'data'
LABELS_KEY = 'watersheds/ws_seeded_z'

# GRAPH_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/graph.n5'
GRAPH_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/graph_A+.n5'

TMP_FOLDER = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/tmp_files_graph'
BLOCK_SHAPE = (25, 256, 256)
EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def jobs_for_cluster_test(n_jobs, n_scales):
    make_batch_jobs(LABELS_PATH, LABELS_KEY,
                    GRAPH_PATH, TMP_FOLDER,
                    BLOCK_SHAPE,
                    n_scales, n_jobs,
                    EXECUTABLE,
                    use_bsub=False,
                    n_threads_merge=4,
                    eta=[5, 5, 5, 5])


if __name__ == '__main__':
    n_jobs = 16
    n_scales = 2
    jobs_for_cluster_test(n_jobs, n_scales)
