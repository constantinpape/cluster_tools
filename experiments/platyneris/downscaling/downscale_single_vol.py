#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import json
from shutil import rmtree

import luigi
import z5py
from cluster_tools.downscaling import DownscalingWorkflow
from cremi_tools.viewer.volumina import view


def downscale_test():
    tmp_folder = './tmp_test'
    config_dir = './configs_test'

    try:
        os.mkdir(config_dir)
    except OSError:
        pass

    config = DownscalingWorkflow.get_config()
    global_config = config['global']
    global_config.update({'shebang': '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'})
    global_config.update({'roi_begin': [50, 500, 500],
                          'roi_end': [150, 1500, 1500]})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    scale_factors = [[1, 2, 2], 2, 2]
    halos = [[0, 10, 10], [10, 10, 10], [10, 10, 10]]

    task = DownscalingWorkflow(tmp_folder=tmp_folder,
                               max_jobs=8,
                               config_dir=config_dir,
                               target='local',
                               input_path='./test.n5',
                               input_key='volumes/raw/s0',
                               output_key_prefix='volumes/raw',
                               scale_factors=scale_factors,
                               halos=halos)
    success = luigi.build([task], local_scheduler=True)

    if success:
        with z5py.File('./test.n5') as f:
            ds = f['volumes/raw/s3']
            ds.n_threads = 8
            raw_ds = ds[:]
            print(raw_ds.shape)
        view([raw_ds])


def get_roi(roi_name):
    rois = {'parapodium': [[5084, 0, 0], [6819, None, None]],
            'block1': [[2607, 11848, 17848], [2993, 15152, 21152]],
            'block2': [[2107, 15848, 21848], [2493, 19152, 25152]],
            'block3': [[2357, 15148, 12348], [2743, 18952, 15652]],
            'block4': [[2007, 14348, 19848], [2393, 17652, 23152]],
            'block5': [[1857, 11848, 15848], [2243, 15152, 19152]],
            'block6': [[1407, 11848, 15648], [1793, 15152, 18952]],
            'block7': [[1957,  9848, 18848], [2343, 13152, 22152]],
            'block8': [[1807, 14348, 13348], [2193, 17652, 16652]],
            'lower_roi': [[0, 0, 0], [5084, None, None]],
            'upper_roi': [[6819, 0, 0], [None, None, None]],
            'full': [[0, 0, 0], [None, None, None]]
           }
    return rois[roi_name]


def downscale_volume(roi_name, max_jobs=250, target='slurm'):
    config_dir = './configs'

    if roi_name is None:
        roi_begin, roi_end = None, None
        tmp_folder = './tmp'
    else:
        roi_begin, roi_end = get_roi(roi_name)
        tmp_folder = './tmp_%s' % roi_name

    try:
        os.mkdir(config_dir)
    except OSError:
        pass

    config = DownscalingWorkflow.get_config()
    global_config = config['global']
    global_config.update({'shebang': '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python',
                          'roi_begin': roi_begin,
                          'roi_end': roi_end})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    task_config = config['downscaling']
    task_config.update({'time_limit': 120,
                        'mem_limit': 3})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(task_config, f)

    scale_factors = [[1, 2, 2], 2, 2, 2, 2]
    halos = [[0, 10, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10]]
    # scale_factors = [[1, 2, 2]]
    # halos = [[0, 10, 10]]

    path = '/g/kreshuk/data/arendt/platyneris_v1/data.n5'

    task = DownscalingWorkflow(tmp_folder=tmp_folder,
                               max_jobs=max_jobs,
                               config_dir=config_dir,
                               target=target,
                               input_path=path,
                               input_key='volumes/raw/s0',
                               output_key_prefix='volumes/raw',
                               scale_factors=scale_factors,
                               halos=halos)
    success = luigi.build([task], local_scheduler=True)
    view_ = False
    if view_ and success and target == 'local':
        sfa = [2, 4, 4]
        roi_begin = tuple(roib // sf for roib, sf in zip(roi_begin, sfa))
        roi_end = tuple(roie // sf for roie, sf in zip(roi_end, sfa))
        bb = tuple(slice(roib, roie) for roib, roie in zip(roi_begin, roi_end))
        print(bb)
        with z5py.File(path) as f:
            ds = f['volumes/raw/s2']
            ds.n_threads = 8
            data = ds[bb]
            print(data.shape)
        view([data])


if __name__ == '__main__':
    downscale_volume('full', max_jobs=100, target='slurm')
    # downscale_test()
    # block_ids = range(1, 9)
    # for block_id in block_ids:
    #     downscale_volume('block%i' % block_id, max_jobs=8, target='local')
