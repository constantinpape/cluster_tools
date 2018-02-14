import os
import sys

EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def make_ws_scripts(path, n_jobs, block_shape):
    sys.path.append('../../..')
    from cluster_tools.masked_watershed import make_batch_jobs
    chunks = [bs // 2 for bs in block_shape]
    # chunks = block_shape
    make_batch_jobs(path, 'predictions/full_affs',
                    path, 'min_filter_mask',
                    path, 'segmentations/watershed',
                    os.path.join(path, 'tmp_files', 'tmp_ws'),
                    block_shape, chunks, n_jobs, EXECUTABLE,
                    use_bsub=True,
                    n_threads_ufd=4,
                    eta=[20, 5, 5, 5])


def make_relabel_scripts(path, n_jobs, block_shape):
    sys.path.append('../../..')
    from cluster_tools.relabel import make_batch_jobs
    make_batch_jobs(path, 'segmentations/watershed',
                    os.path.join(path, 'tmp_files', 'tmp_relabel'),
                    block_shape, n_jobs,
                    EXECUTABLE,
                    use_bsub=True,
                    eta=[5, 5, 5])


def make_graph_scripts(path, n_scales, n_jobs, n_threads, block_shape):
    sys.path.append('../../..')
    from cluster_tools.graph import make_batch_jobs
    make_batch_jobs(path, 'segmentations/watershed',
                    os.path.join(path, 'tmp_files', 'graph.n5'),
                    os.path.join(path, 'tmp_files', 'tmp_graph'),
                    block_shape,
                    n_scales, n_jobs,
                    EXECUTABLE,
                    use_bsub=True,
                    n_threads_merge=n_threads,
                    eta=[10, 10, 10, 10])


def make_feature_scripts(path, n_jobs1, n_jobs2, n_threads, block_shape):
    sys.path.append('../../..')
    from cluster_tools.features import make_batch_jobs
    make_batch_jobs(os.path.join(path, 'tmp_files', 'graph.n5'), 'graph',
                    os.path.join(path, 'tmp_files', 'features.n5'), 'features',
                    path, 'predictions/full_affs',
                    path, 'segmentations/watershed',
                    os.path.join(path, 'tmp_files', 'tmp_features'),
                    block_shape,
                    n_jobs1, n_jobs2,
                    n_threads2=n_threads,
                    executable=EXECUTABLE,
                    use_bsub=True,
                    eta=[20, 5])


def make_multicut_scripts(path, n_scales, n_jobs, n_threads, block_shape):
    sys.path.append('../../..')
    from cluster_tools.multicut import make_batch_jobs
    make_batch_jobs(os.path.join(path, 'tmp_files', 'graph.n5'), 'graph',
                    os.path.join(path, 'tmp_files', 'features.n5'), 'features',
                    path, 'node_labelings/multicut',
                    block_shape, n_scales,
                    os.path.join(path, 'tmp_files', 'tmp_mc'),
                    n_jobs,
                    n_threads=n_threads,
                    executable=EXECUTABLE,
                    use_bsub=False,
                    eta=[5, 5, 15])


def make_projection_scripts(path, n_jobs, block_shape):
    sys.path.append('../../..')
    from cluster_tools.label_projection import make_batch_jobs
    chunks = [bs // 2 for bs in block_shape]
    # chunks = block_shape
    make_batch_jobs(path, 'segmentations/watershed',
                    path, 'segmentations/multicut',
                    path, 'node_labelings/multicut',
                    os.path.join(path, 'tmp_files', 'tmp_projection'),
                    block_shape, chunks, n_jobs,
                    executable=EXECUTABLE,
                    use_bsub=False,
                    eta=5)


def make_scripts(path,
                 n_scales,
                 n_jobs_max,
                 n_threads_max,
                 block_shape):
    # make folders
    if not os.path.exists(os.path.join(path, 'segmentations')):
        os.mkdir(os.path.join(path, 'segmentations'))
    if not os.path.exists(os.path.join(path, 'node_labelings')):
        os.mkdir(os.path.join(path, 'node_labelings'))
    if not os.path.exists(os.path.join(path, 'tmp_files')):
        os.mkdir(os.path.join(path, 'tmp_files'))

    # make the ws scripts
    if not os.path.exists('./1_watershed'):
        os.mkdir('./1_watershed')
    os.chdir('./1_watershed')
    make_ws_scripts(path, n_jobs_max, block_shape)
    os.chdir('..')

    # make the relabeling scripts
    if not os.path.exists('./2_relabel'):
        os.mkdir('./2_relabel')
    os.chdir('./2_relabel')
    make_relabel_scripts(path, n_jobs_max, block_shape)
    os.chdir('..')

    # make the graph scripts
    if not os.path.exists('./3_graph'):
        os.mkdir('./3_graph')
    os.chdir('./3_graph')
    make_graph_scripts(path, n_scales, n_jobs_max, n_threads_max, block_shape)
    os.chdir('..')

    # make the feature scripts
    if not os.path.exists('./4_features'):
        os.mkdir('./4_features')
    os.chdir('./4_features')
    make_feature_scripts(path, n_jobs_max, 4, n_threads_max, block_shape)
    os.chdir('..')

    # make the multicut scripts
    if not os.path.exists('./5_multicut'):
        os.mkdir('./5_multicut')
    os.chdir('./5_multicut')
    make_multicut_scripts(path, n_scales, n_jobs_max, n_threads_max, block_shape)
    os.chdir('..')

    # make the projection scripts
    if not os.path.exists('./6_label_projection'):
        os.mkdir('./6_label_projection')
    os.chdir('./6_label_projection')
    make_projection_scripts(path, n_jobs_max, block_shape)
    os.chdir('..')


if __name__ == '__main__':
    path = '/nrs/saalfeld/lauritzen/01/workspace.n5/raw'
    n_jobs = 100
    n_scales = 1
    n_threads = 32
    block_shape = (50, 512, 512)
    make_scripts(path, n_scales, n_jobs, n_threads, block_shape)
