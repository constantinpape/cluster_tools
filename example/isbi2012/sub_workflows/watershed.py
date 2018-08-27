#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi
import z5py

from cluster_tools.watershed import WatershedWorkflow
from cluster_tools.watershed.watershed import WatershedLocal


def ws_example(shebang):
    example_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'

    input_key = 'volumes/affinities'
    output_key = 'volumes/ws'

    max_jobs = 8

    global_conf = WatershedLocal.default_global_config()
    global_conf.update({'shebang': shebang})
    try:
        os.mkdir('configs')
    except OSError:
        pass

    with open('./configs/global.config', 'w') as f:
        json.dump(global_conf, f)

    ret = luigi.build([WatershedWorkflow(input_path=example_path, input_key=input_key,
                                         output_path=example_path, output_key=output_key,
                                         config_dir='./configs',
                                         tmp_folder='./tmp',
                                         target='local',
                                         max_jobs=max_jobs)], local_scheduler=True)
    if ret:
        from cremi_tools.viewer.volumina import view
        with z5py.File(example_path) as f:
            affs = f[input_key][:3].transpose((1, 2, 3, 0))
            ws = f[output_key][:]
        view([affs, ws])


if __name__ == '__main__':
    shebang = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
    ws_example(shebang)
