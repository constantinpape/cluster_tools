#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
import json
import luigi
from cluster_tools.ilastik import IlastikPredictionWorkflow


def run_pred(tmp_folder, max_jobs, max_threads, target):

    input_path = '/g/kreshuk/data/arendt/sponge/prediction/data.h5'
    input_key = 'data'

    output_path = '/g/kreshuk/data/arendt/sponge/data.n5'
    output_key = 'volumes/predictions/semantic_stage1'

    configs = IlastikPredictionWorkflow.get_config()

    config_folder = 'config'
    if not os.path.exists(config_folder):
        os.mkdir(config_folder)

    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python"
    global_config = configs['global']
    global_config.update({'shebang': shebang,
                          'block_shape': (512, 512, 512)})
    with open('./config/global.config', 'w') as f:
        json.dump(global_config, f)

    conf = configs['prediction']
    conf.update({'time_limit': 1440, 'mem_limit': 16, 'threads_per_job': max_threads,
                 'qos': 'high'})
    with open('./config/prediction.config', 'w') as f:
        json.dump(conf, f)

    # conf = configs['merge_predictions']
    # conf.update({'time_limit': 1440, 'mem_limit': 24})
    # with open('./config/merge_predictions.config', 'w') as f:
    #     json.dump(conf, f)

    ilastik_project = '/g/kreshuk/data/arendt/sponge/'
    ilastik_folder = '/g/kreshuk/software/ilastik-1.3.2rc2-Linux'
    halo = [32, 32, 32]
    n_channels = 6

    ret = luigi.build([IlastikPredictionWorkflow(input_path=input_path, input_key=input_key,
                                                 output_path=output_path, output_key=output_key,
                                                 ilastik_project=ilastik_project,
                                                 ilastik_folder=ilastik_folder,
                                                 halo=halo, n_channels=n_channels,
                                                 config_dir=config_folder,
                                                 tmp_folder=tmp_folder,
                                                 target=target,
                                                 max_jobs=max_jobs)], local_scheduler=True)
    assert ret


if __name__ == '__main__':
    tmp_folder = './tmp'
    max_jobs = 200
    max_threads = 4
    target = 'slurm'

    # max_jobs = 2
    # target = 'local'

    run_pred(tmp_folder, max_jobs, max_threads, target)
