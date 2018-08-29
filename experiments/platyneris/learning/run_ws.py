#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi
import h5py
import z5py

from cluster_tools.watershed import WatershedWorkflow


def run_ws(block_id, max_jobs,
           target='local'):

    path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/predictions/train_block_0%i_unet_lr_v3.n5' % block_id
    input_key = 'volumes/affinities'

    configs = WatershedWorkflow.get_config()
    ws_config = configs['watershed']

    ws_config.update({'threshold': .25, 'apply_presmooth_2d': False,
                      'sigma_weights': (1., 2., 2.), 'apply_dt_2d': False,
                      'pixel_pitch': (2, 1, 1), 'two_pass': True,
                      'halo': [0, 50, 50]})
    with open('./config/watershed.config', 'w') as f:
        json.dump(ws_config ,f)

    tmp_folder = './tmp_%i' % block_id
    ret = luigi.build([WatershedWorkflow(input_path=path, input_key=input_key,
                                         output_path=path, output_key='volumes/watershed',
                                         config_dir='./config',
                                         tmp_folder=tmp_folder,
                                         target=target,
                                         max_jobs=max_jobs)], local_scheduler=True)

if __name__ == '__main__':

    block_id = int(sys.argv[1])
    # target = 'slurm'
    # max_jobs = 32

    target = 'local'
    max_jobs = 8

    run_ws(block_id, max_jobs, target=target)
