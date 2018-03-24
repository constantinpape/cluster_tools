import os
import sys
import hashlib

EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def make_cc_scripts(path, n_jobs, block_shape, tmp_dir):
    sys.path.append('../../..')
    from cluster_tools.connected_components import make_batch_jobs
    chunks = block_shape
    # chunks = block_shape
    make_batch_jobs(path, 'syncleft_dist_DTU-2_200000',
                    path, 'syncleft_dist_DTU-2_200000_cc',
                    os.path.join(tmp_dir, 'tmp_files', 'tmp_cc'),
                    block_shape, chunks, n_jobs, EXECUTABLE,
                    use_bsub=True,
                    n_threads_ufd=4,
                    eta=[20, 5, 5, 5])


def make_scripts(path,
                 n_jobs,
                 block_shape,
                 tmp_dir):
    # make folders
    if not os.path.exists(os.path.join(path, 'segmentations')):
        os.mkdir(os.path.join(path, 'segmentations'))
    if not os.path.exists(os.path.join(path, 'node_labelings')):
        os.mkdir(os.path.join(path, 'node_labelings'))
    if not os.path.exists(os.path.join(tmp_dir, 'tmp_files')):
        os.mkdir(os.path.join(tmp_dir, 'tmp_files'))

    # make the minfilter scripts
    if not os.path.exists('./ccs'):
        os.mkdir('./ccs')
    os.chdir('./ccs')
    make_cc_scripts(path, n_jobs, block_shape, tmp_dir)
    os.chdir('..')


if __name__ == '__main__':
    path = '/nrs/saalfeld/lauritzen/02/workspace.n5'
    mhash = hashlib.md5(path.encode('utf-8')).hexdigest()
    tmp_dir = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/scotts_block_%s' % mhash
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    n_jobs = 200
    block_shape = (50, 512, 512)
    make_scripts(path, n_jobs, block_shape, tmp_dir)
