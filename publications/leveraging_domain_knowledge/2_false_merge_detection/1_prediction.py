#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch/bin/python

import os
import json
import luigi
from cluster_tools.inference import InferenceLocal, InferenceSlurm


def predict(device_mapping, target):
    max_jobs = len(device_mapping)
    tmp_folder = './tmp_inference'
    input_path = '/g/kreshuk/data/FIB25/data.n5'
    output_path = input_path
    roi_begin = roi_end = None

    in_key = 'volumes/raw/s0'
    out_key = {'volumes/affinities/s0': (0, 3)}

    mask_path = input_path
    mask_key = 'volumes/masks/minfilter/s5'

    config_folder = './config_inference'
    if not os.path.exists(config_folder):
        os.mkdir(config_folder)

    input_blocks = (192,) * 3
    # remove (16, 16, 16) pixels from each side in the output
    output_blocks = (160,) * 3
    halo = [(ib - ob) // 2 for ib, ob in zip(input_blocks, output_blocks)]
    print("Found halo", halo)

    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch/bin/python"
    global_config = InferenceLocal.default_global_config()
    global_config.update({'shebang': shebang,
                          'block_shape': output_blocks,
                          'roi_begin': roi_begin,
                          'roi_end': roi_end})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    config = InferenceLocal.default_task_config()
    config.update({'chunks': [ob // 2 for ob in output_blocks],
                   'mem_limit': 32, 'time_limit': 720,
                   'threads_per_job': 4, 'set_visible_device': False})
    with open(os.path.join(config_folder, 'inference.config'), 'w') as f:
        json.dump(config, f)

    ckpt = '/g/kreshuk/matskevych/boundary_map_prediction/project_folder_old/Weights'
    task = InferenceLocal if target == 'local' else InferenceSlurm
    t = task(tmp_folder=tmp_folder, max_jobs=max_jobs,
             config_dir=config_folder,
             input_path=input_path, input_key=in_key,
             output_path=output_path, output_key=out_key,
             mask_path=mask_path, mask_key=mask_key,
             checkpoint_path=ckpt, framework='inferno',
             halo=halo)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Failure"


if __name__ == '__main__':
    target = 'local'
    device_mapping = {}
    predict(device_mapping, target)
