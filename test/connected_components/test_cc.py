import sys
sys.path.append('../..')
from cluster_tools.connected_components import make_batch_jobs_step1

IN_PATH = './binary_volume.n5'
IN_KEY = 'data'
OUT_PATH = './ccs.n5'
OUT_KEY = 'data'
TMP_FOLDER = './tmp'
BLOCK_SHAPE = (50, 500, 500)
CHUNKS = (5, 250, 250)
EXECUTABLE = '/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def step1(n_jobs):
    make_batch_jobs_step1(IN_PATH, IN_KEY, OUT_PATH, OUT_KEY, TMP_FOLDER,
                          BLOCK_SHAPE, CHUNKS, n_jobs, EXECUTABLE, use_bsub=False)


if __name__ == '__main__':
    step1(n_jobs=4)
