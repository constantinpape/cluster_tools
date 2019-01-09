#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import numpy as np
import luigi
import z5py
import h5py

from cluster_tools.connected_components import ConnectedComponentsWorkflow
from cremi_tools.viewer.volumina import view


def write_max_id():
    input_path = '/g/kreshuk/data/cremi/original/sample_A_20160501.hdf'
    input_key = 'volumes/labels/neuron_ids'
    with h5py.File(input_path) as f:
        seg = f[input_key][:]
        max_id = int(seg.max())
        f[input_key].attrs['maxId'] = max_id


def run_ccs(max_jobs, target='local'):

    # input_path = '/home/cpape/Work/data/cremi/sample_A_20160501.hdf'
    input_path = '/g/kreshuk/data/cremi/original/sample_A_20160501.hdf'
    input_key = 'volumes/labels/neuron_ids'
    output_path = './sampleA.n5'
    output_key = 'volumes/labels/neuron_ids_cc'

    configs = ConnectedComponentsWorkflow.get_config()

    config_folder = 'config_ccs'
    if not os.path.exists(config_folder):
        os.mkdir(config_folder)

    # shebang = "#! /home/cpape/Work/software/conda/miniconda3/envs/main/bin/python"
    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python"
    global_config = configs['global']
    global_config.update({'shebang': shebang})
    with open('./config_ccs/global.config', 'w') as f:
        json.dump(global_config, f)

    tmp_folder = './tmp_ccs'
    ret = luigi.build([ConnectedComponentsWorkflow(input_path=input_path, input_key=input_key,
                                                   output_path=output_path, output_key=output_key,
                                                   config_dir=config_folder,
                                                   tmp_folder=tmp_folder,
                                                   target=target,
                                                   max_jobs=max_jobs)], local_scheduler=True)
    if ret:
        with h5py.File(input_path) as f:
            raw = f['volumes/raw'][:]
            seg = f['volumes/labels/neuron_ids'][:]
        with z5py.File(output_path) as f:
            seg_cc = f['volumes/labels/neuron_ids_cc'][:]
        view([raw, seg, seg_cc])


if __name__ == '__main__':
    target = 'local'
    max_jobs = 8
    # write_max_id()
    run_ccs(max_jobs, target=target)
