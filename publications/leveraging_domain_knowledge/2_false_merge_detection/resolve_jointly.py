import os
import json
import luigi
from shutil import copyfile

from cluster_tools import LiftedMulticutSegmentationWorkflow


# TODO once we have proper validation set up, we should check for the
# influence of the multicut block-shape (esp. comparing it to overlapping / not-overlapping watersheds)
# plant experiments show that this might change a lot
# probably need to implement some new features in cluster_tools for this
def resolve_jointly(identifier, max_jobs=48, target='local'):
    task = LiftedMulticutSegmentationWorkflow

    input_path = '/g/kreshuk/data/FIB25/data.n5'
    exp_path = '/g/kreshuk/data/FIB25/exp_data/%s.n5' % identifier

    assignment_out_key = 'node_labels/resolve_jointly/%s' % identifier
    ws_key = 'volumes/segmentation/watershed'
    out_key = 'volumes/segmentation/resolve_jointly/%s' % identifier
    input_key = 'volumes/affinities'
    mask_key = 'volumes/masks/minfilter/s5'

    tmp_folder = './tmp_folders/tmp_resolve_jointly_%s' % identifier
    os.makedirs(tmp_folder, exist_ok=True)
    # make log for costs in order to skip all the problem computation
    copyfile('./tmp_mc/probs_to_costs.log',
             os.path.join(tmp_folder, 'probs_to_costs.log'))

    config_dir = './config_mc'
    configs = task.get_config()
    conf_names = ['solve_lifted_subproblems',
                  'solve_lifted_global',
                  'reduce_lifted_problem']
    for name in conf_names:
        conf = configs[name]
        conf.update({'threads_per_job': max_jobs})
        with open(os.path.join(config_dir, '%s.config' % name), 'w') as f:
            json.dump(conf, f)

    n_scales = 1
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target, max_jobs_multicut=1,
             input_path=input_path, input_key=input_key,
             mask_path=input_path, mask_key=mask_key,
             ws_path=input_path, ws_key=ws_key,
             problem_path=exp_path,
             node_labels_key=assignment_out_key,
             output_path=input_path, output_key=out_key,
             lifted_labels_path='', lifted_labels_key='',
             lifted_prefix='resolving',
             n_scales=n_scales, skip_ws=True)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Resolving failed"


if __name__ == '__main__':
    resolve_jointly('perfect_oracle')
