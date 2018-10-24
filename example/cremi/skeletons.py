#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import json
from shutil import rmtree

import numpy as np
import luigi
import z5py
from cluster_tools.skeletons import SkeletonWorkflow
from cremi_tools.viewer.volumina import view


def skeletons(sample, max_jobs=8, target='local'):
    input_path = '/g/kreshuk/data/cremi/realigned/sample%s_small.n5' % sample
    input_prefix = 'segmentation/multicut'
    output_prefix = 'skeletons/multicut'

    config_dir = './config_skeletons'
    tmp_folder = './tmp_skeletons_%s' % sample

    try:
        os.mkdir(config_dir)
    except OSError:
        pass

    config = SkeletonWorkflow.get_config()
    global_config = config['global']
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    ds_config = config['skeletonize']
    ds_config.update({'threads_per_job': 8})
    with open(os.path.join(config_dir, 'skeletonize.config'), 'w') as f:
        json.dump(ds_config, f)

    task = SkeletonWorkflow(tmp_folder=tmp_folder,
                            max_jobs=1,
                            config_dir=config_dir,
                            target='local',
                            input_path=input_path,
                            output_path=input_path,
                            input_prefix=input_prefix,
                            output_prefix=output_prefix,
                            work_scale=2)
    success = luigi.build([task], local_scheduler=True)

    #
    if success and target == 'local':
        with z5py.File(input_path) as f:
            #
            ds = f['skeletons/multicut/s2']
            ds.n_threads = 8
            skels = ds[:]

            #
            ds = f['raw/s2']
            ds.n_threads = 8
            raw = ds[:]

            #
            ds = f['segmentation/multicut/s2']
            ds.n_threads = 8
            seg = ds[:]


        view([raw, seg, skels], ['raw', 'seg', 'skels'])


if __name__ == '__main__':
    skeletons('A')
