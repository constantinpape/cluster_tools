import os
import json
import z5py
import luigi
from cluster_tools.paintera import ConversionWorkflow


def to_paintera_format():
    target = 'local'
    max_jobs = 8

    path = './data/data.n5'
    with z5py.File(path) as f:
        ds = f['labels']
        offset = ds.attrs['offset']
        resolution = ds.attrs['resolution']

    config_dir = './configs'
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    configs = ConversionWorkflow.get_config()

    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/paintera/bin/python"
    global_config = configs['global']
    global_config.update({'shebang': shebang})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_config, f)

    sampling_config = {'library': 'vigra', 'library_kwargs': {'order': 0}}

    ds_config = configs['downscaling']
    ds_config.update({**sampling_config})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(ds_config, f)

    up_config = configs['upscaling']
    up_config.update({**sampling_config})
    with open(os.path.join(config_dir, 'upscaling.config'), 'w') as f:
        json.dump(up_config, f)

    t = ConversionWorkflow(path=path,
                           raw_key='volumes/raw',
                           label_in_key='labels',
                           label_out_key='volumes/labels/neuron_ids',
                           assignment_key='',
                           label_scale=0,
                           offset=offset,
                           resolution=resolution,
                           tmp_folder='./tmp',
                           max_jobs=max_jobs,
                           config_dir=config_dir,
                           target=target)
    luigi.build([t], local_scheduler=True)


if __name__ == '__main__':
    to_paintera_format()
