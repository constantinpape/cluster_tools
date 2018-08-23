#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi
import h5py

from cluster_tools.watershed import WatershedWorkflow
from cluster_tools.watershed.watershed import WatershedBase


def run_ws(input_path, output_path, input_key, output_key, tmp_folder, max_jobs,
           target='slurm'):

    config = WatershedBase.default_task_config()
    config.update({'threshold': .25, 'apply_presmooth_2d': False,
                   'sigma_weights': (1., 2., 2.), 'apply_dt_2d': False,
                   'pixel_pitch': (2, 1, 1), 'two_pass': True,
                   'halo': [0, 50, 50]})
    with open('./config/watershed.config', 'w') as f:
        json.dump(config ,f)

    ret = luigi.build([WatershedWorkflow(input_path=input_path, input_key=input_key,
                                         output_path=output_path, output_key=output_key,
                                         config_dir='./config',
                                         tmp_folder=tmp_folder,
                                         target=target,
                                         max_jobs=max_jobs)], local_scheduler=True)


if __name__ == '__main__':
    input_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/predictions/val_block_01_unet_lr_v3_global_norm.h5'
    output_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/segmentation/val_block_01_unet_lr_v3_global_norm_ws.n5'
    input_key = output_key = 'data'

    with h5py.File(input_path, 'r') as f:
        assert input_key in f

    tmp_folder = '/g/kreshuk/pape/Work/data/cache/plat_val'
    max_jobs = 32

    run_ws(input_path, output_path,
           input_key, output_key,
           tmp_folder, max_jobs)
