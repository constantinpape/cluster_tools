#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import json
from shutil import rmtree

import luigi
import z5py
from cluster_tools.downscaling import DownscalingWorkflow
from cremi_tools.viewer.volumina import view


def downscale_raw(sample, max_jobs=8, target='local'):
    input_path = '/g/kreshuk/data/cremi/realigned/sample%s_small.n5' % sample
    input_key = 'raw'

    config_dir = './config_ds_raw'
    tmp_folder = './tmp_ds_raw_%s' % sample

    try:
        os.mkdir(config_dir)
    except OSError:
        pass

    config = DownscalingWorkflow.get_config()
    global_config = config['global']
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    task_config = config['downscaling']
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(task_config, f)

    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], 2]
    halos = [[0, 10, 10], [0, 10, 10], [0, 10, 10], [10, 10, 10]]

    task = DownscalingWorkflow(tmp_folder=tmp_folder,
                               max_jobs=max_jobs,
                               config_dir=config_dir,
                               target=target,
                               input_path=input_path,
                               input_key='raw/s0',
                               output_key_prefix='raw',
                               scale_factors=scale_factors,
                               halos=halos)
    success = luigi.build([task], local_scheduler=True)
    #
    if success and target == 'local':
        with z5py.File(input_path) as f:
            ds = f['volumes/raw/s2']
            ds.n_threads = 8
            data = ds[:]
            print(data.shape)
        view([data])


def downscale_seg(sample, max_jobs=8, target='local'):
    input_path = '/g/kreshuk/data/cremi/realigned/sample%s_small.n5' % sample
    input_key = 'segmentation/multicut/s0'

    config_dir = './config_ds_seg'
    tmp_folder = './tmp_ds_seg_%s' % sample

    try:
        os.mkdir(config_dir)
    except OSError:
        pass

    config = DownscalingWorkflow.get_config()
    global_config = config['global']
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    ds_config = config['downscaling']
    # FIXME majority vote downscaling is broken
    # ds_config.update({'library': 'skimage', 'threads_per_job': 8})
    ds_config.update({'library': 'vigra', 'library_kwargs': {'order': 0}, 'threads_per_job': 8})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(ds_config, f)

    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], 2]
    halos = [[0, 10, 10], [0, 10, 10], [0, 10, 10], [10, 10, 10]]

    task = DownscalingWorkflow(tmp_folder=tmp_folder,
                               max_jobs=1,
                               config_dir=config_dir,
                               target='local',
                               input_path=input_path,
                               input_key='segmentation/multicut/s0',
                               output_key_prefix='segmentation/multicut',
                               scale_factors=scale_factors,
                               halos=halos)
    success = luigi.build([task], local_scheduler=True)

    #
    if success and target == 'local':
        with z5py.File(input_path) as f:
            #
            ds = f['raw/s2']
            ds.n_threads = 8
            raw = ds[:]
            rshape = raw.shape
            #
            ds = f['segmentation/multicut/s2']
            ds.n_threads = 8
            seg = ds[:]
            mshape = seg.shape
            assert mshape == rshape, "%s %s" % (str(mshape), str(rshape))

        view([raw, seg])


if __name__ == '__main__':
    # downscale_raw('A')
    downscale_seg('A')
