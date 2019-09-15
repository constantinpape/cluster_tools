import os
import json
from shutil import copyfile

import z5py
import luigi
from cluster_tools import LiftedMulticutSegmentationWorkflow

SOLVERS = ('kernighan-lin', 'greedy-additive', 'fusion-moves', 'hierarchical')


def prepare_data(exp_path, tmp_folder, prefix):
    # make log for costs in order to skip all the problem computation
    os.makedirs(tmp_folder, exist_ok=True)
    copyfile('./tmp_mc/probs_to_costs.log',
             os.path.join(tmp_folder, 'probs_to_costs.log'))

    # make all relevant links in exp-data
    scale_folder = os.path.join(exp_path, 's0')
    os.makedirs(scale_folder, exist_ok=True)

    root = '/g/kreshuk/pape/Work/my_projects/lifted_priors/lmc_experiments/exp_data/exp_data.n5'
    # copy shape attribute
    with z5py.File(root) as f:
        shape = f.attrs['shape']
    with z5py.File(exp_path) as f:
        f.attrs['shape'] = shape

    root = os.path.join(root, 's0')
    for name in ('costs', 'graph', 'sub_graphs'):
        os.symlink(os.path.join(root, name),
                   os.path.join(scale_folder, name))
    for name in ('lifted_costs', 'lifted_nh'):
        os.symlink(os.path.join(root, name),
                   os.path.join(scale_folder, '%s_%s' % (name, prefix)))


# run lmc with different solvers:
# - gaec
# - gaec + kernighan lin
# - gaec + kernighan lin + fusion moves(gaec + kernighan lin)
# - blockwise(gaec + kernighan lin)
def run_lmc(solver, target, max_jobs):
    assert solver in SOLVERS
    task = LiftedMulticutSegmentationWorkflow

    input_path = '/g/kreshuk/data/FIB25/cutout.n5'
    exp_path = './exp_data/%s.n5' % solver
    tmp_folder = './tmp_folders/%s' % solver
    prepare_data(exp_path, tmp_folder, solver)

    assignment_out_key = 'node_labels/%s' % solver
    ws_key = 'volumes/segmentation/watershed'
    out_key = 'volumes/segmentation/%s' % solver
    input_key = 'volumes/affinities'

    if solver == 'hierarchical':
        n_scales = 1
        agglomerator = 'kernighan-lin'
    else:
        n_scales = 0
        agglomerator = solver

    config_dir = './config'
    configs = task.get_config()
    conf_names = ['solve_lifted_subproblems',
                  'solve_lifted_global',
                  'reduce_lifted_problem']
    for name in conf_names:
        conf = configs[name]
        conf.update({'threads_per_job': max_jobs, 'agglomerator': agglomerator})
        with open(os.path.join(config_dir, '%s.config' % name), 'w') as f:
            json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target, max_jobs_multicut=1,
             input_path=input_path, input_key=input_key,
             ws_path=input_path, ws_key=ws_key,
             problem_path=exp_path,
             node_labels_key=assignment_out_key,
             output_path=input_path, output_key=out_key,
             lifted_labels_path='', lifted_labels_key='',
             lifted_prefix=solver,
             n_scales=n_scales, skip_ws=True)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, solver


if __name__ == '__main__':
    # solver = 'hierarchical'
    for solver in ('fusion-moves',):
        run_lmc(solver, 'local', 8)
