#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import json
from shutil import rmtree

import luigi
import h5py
from cluster_tools.downscaling import DownscalingWorkflow
from cremi_tools.viewer.volumina import view


def downscale_test():
    tmp_folder = './tmp_bdv'
    config_dir = './configs_bdv'

    try:
        os.mkdir(config_dir)
    except OSError:
        pass

    config = DownscalingWorkflow.get_config()
    global_config = config['global']
    global_config.update({'shebang': '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    ds_config = config['downscaling']
    ds_config.update({'threads_per_job': 8})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(ds_config, f)

    scale_factors = [2, 2, 2]
    halos = [[10, 10, 10], [10, 10, 10], [10, 10, 10]]

    metadata = {'unit': 'micrometer',
                'resolution': (.25, .2, .2),
                'offsets': (0., 0., 0.)}

    task = DownscalingWorkflow(tmp_folder=tmp_folder,
                               max_jobs=1,
                               config_dir=config_dir,
                               target='local',
                               input_path='./test.h5',
                               input_key='volumes/raw',
                               output_path='./test.h5',
                               scale_factors=scale_factors,
                               halos=halos,
                               metadata_format='bdv',
                               metadata_dict=metadata)
    success = luigi.build([task], local_scheduler=True)

    if success:
        with h5py.File('./test.h5') as f:
            ds = f['t00000/s00/3/cells']
            raw_ds = ds[:]
            print(raw_ds.shape)
        view([raw_ds])


def downscale_predictions(max_jobs=16, target='slurm'):
    path = '/g/arendt/EM_6dpf_segmentation/EM-Prospr/em-segmented-membranes-parapodium.h5'
    tmp_folder = './tmp_bdv_parapodium'
    config_dir = './configs_bdv_parapodium'

    try:
        os.mkdir(config_dir)
    except OSError:
        pass

    config = DownscalingWorkflow.get_config()
    global_config = config['global']
    global_config.update({'shebang': '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    ds_config = config['downscaling']
    ds_config.update({'threads_per_job': max_jobs})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(ds_config, f)

    n_scales = 5
    scale_factors = n_scales * [[2, 2, 2]]
    halos = n_scales * [[10, 10, 10]]

    resolution = (.25, .2, .2)
    offsets = (5084, 0, 0)

    # transfer the offsets from measure in pixels to measure in
    # micrometer
    offsets = tuple(off * res for off, res in zip(offsets, resolution))

    metadata = {'unit': 'micrometer',
                'resolution': resolution,
                'offsets': offsets}

    task = DownscalingWorkflow(tmp_folder=tmp_folder,
                               max_jobs=1,
                               config_dir=config_dir,
                               target=target,
                               input_path=path,
                               input_key='boundary_channel',
                               output_path=path,
                               scale_factors=scale_factors,
                               halos=halos,
                               metadata_format='bdv',
                               metadata_dict=metadata)
    success = luigi.build([task], local_scheduler=True)

    if success:
        with h5py.File('./test.h5') as f:
            ds = f['t00000/s00/3/cells']
            raw_ds = ds[:]
            print(raw_ds.shape)
        view([raw_ds])


if __name__ == '__main__':
    downscale_predictions()
    # downscale_test()
