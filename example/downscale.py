#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
import json
import sys
import luigi
from cluster_tools.downscaling import DownscalingWorkflow


def downscale_raw(path, max_jobs=8, target='local'):
    """ Downscale raw data.

    Arguments:
        path [str] - path to raw data
        max_jobs [int] - maximum number of jobs
        target [str] - target of computation: local, slurm or lsf
    """

    # input and output keys
    input_key = 'raw'
    output_key = 'volumes/raw'

    # temporary directories
    config_dir = './configs'
    tmp_folder = './tmp_downscaling'
    os.makedirs(config_dir, exist_ok=True)

    # write the global configiration with shebang of python env with
    # all necessary dependencies
    config = DownscalingWorkflow.get_config()
    global_config = config['global']
    shebang = f'#! {sys.executable}'
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    # write the task specific config
    # here, we configure the downscaling task to use skimage
    task_config = config['downscaling']
    task_config.update({'library': 'skimage'})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(task_config, f)

    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], 2]
    halos = [[0, 10, 10], [0, 10, 10], [0, 10, 10], [10, 10, 10]]

    task = DownscalingWorkflow(tmp_folder=tmp_folder,
                               max_jobs=max_jobs,
                               config_dir=config_dir,
                               target=target,
                               input_path=path,
                               input_key=input_key,
                               output_key_prefix=output_key,
                               scale_factors=scale_factors,
                               halos=halos)
    success = luigi.build([task], local_scheduler=True)
    assert success, "Dowscaling failed"


if __name__ == '__main__':
    # path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    path = "./sampleA.n5"
    downscale_raw(path, max_jobs=8, target='local')
