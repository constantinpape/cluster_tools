import argparse
import json
import multiprocessing
import os

import luigi
from cluster_tools import MulticutSegmentationWorkflow


def run_multicut(target, n_jobs):
    task = MulticutSegmentationWorkflow

    input_path = './data.n5'
    tmp_folder = './tmp_mc'
    exp_path = os.path.join(tmp_folder, 'exp_data.n5')
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    input_key = 'probs/membranes'
    ws_key = 'volumes/segmentation/watershed'
    out_key = 'volumes/segmentation/multicut'
    node_labels_key = 'node_labels/multicut'

    configs = task.get_config()

    global_config = configs['global']
    global_config.update({'block_shape': [140, 140, 140]})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    ws_config = configs['watershed']
    ws_config.update({'threshold': .3, 'apply_presmooth_2d': False,
                      'sigma_weights': 2., 'apply_dt_2d': False,
                      'sigma_seeds': 2., 'apply_ws_2d': False,
                      'two_pass': False, 'alpha': .5,
                      'halo': [25, 25, 25], 'size_filter': 100})
    with open(os.path.join(config_folder, 'watershed.config'), 'w') as f:
        json.dump(ws_config, f)

    t = task(input_path=input_path, input_key=input_key,
             ws_path=input_path, ws_key=ws_key,
             problem_path=exp_path, node_labels_key=node_labels_key,
             output_path=input_path, output_key=out_key,
             config_dir=config_folder, tmp_folder=tmp_folder,
             target=target, max_jobs=n_jobs, n_scales=1)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Multicut segmentation failed"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='local', type=str)
    parser.add_argument('--n_jobs', default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()
    run_multicut(args.target, args.n_jobs)
