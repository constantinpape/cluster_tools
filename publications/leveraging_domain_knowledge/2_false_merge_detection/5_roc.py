import os
import json
from shutil import rmtree
from shutil import copyfile

import numpy as np
import luigi

from cluster_tools import LiftedMulticutSegmentationWorkflow
from resolving import ResolvingWorkflow, LiftedEdgesLocal, LiftedEdgesSlurm
from evaluate import evaluate_fib, CopyAndCropLocal, CopyAndCropSlurm

# ROI from Michals mail
ROI_START = [5074, 1600, 1800]
ROI_SIZE = [2876, 2401, 2901]


def copy_and_crop_seg(seg_path, seg_key, tmp_path, tmp_key, config_dir,
                      tmp_folder, target, max_threads):
    copy_task = CopyAndCropSlurm if target == 'slurm' else CopyAndCropLocal

    # copy and crop
    copy_conf = copy_task.default_task_config()
    copy_conf.update({'threads_per_job': max_threads,
                      'mem_limit': 256})
    with open(os.path.join(config_dir, 'copy_and_crop.config'), 'w') as f:
        json.dump(copy_conf, f)

    t = copy_task(tmp_folder=tmp_folder, config_dir=config_dir,
                  max_jobs=1,
                  input_path=seg_path, input_key=seg_key,
                  output_path=tmp_path, output_key=tmp_key,
                  roi_start=ROI_START, roi_size=ROI_SIZE)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def lifted_problem(tmp_folder, config_dir,
                   target, max_threads,
                   path, exp_path, recall, precision):
    task = LiftedEdgesSlurm if target == 'slurm' else LiftedEdgesLocal

    config = task.default_task_config()
    config.update({'threads_per_job': max_threads})
    with open(os.path.join(config_dir, 'lifted_edges.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir, max_jobs=1,
             path=path, exp_path=exp_path, precision=precision, recall=recall)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def solve_separately(path, exp_path, tmp_folder, target, max_jobs, n_threads):
    task = ResolvingWorkflow

    ws_key = 'volumes/segmentation/watershed'
    seg_key = 'volumes/segmentation/separately'
    objects_group = 'resolving/oracle'

    assignment_in_key = 'node_labels/multitcut_filtered'
    assignment_out_key = 'node_labels/separately'

    config_dir = './configs'
    conf = task.get_config()['resolve_inidividual_objects']
    conf.update({'threads_per_job': n_threads, 'mem_limit': 128})
    with open(os.path.join(config_dir, 'resolve_inidividual_objects.config'), 'w') as f:
        json.dump(conf, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             problem_path=exp_path, path=path,
             output_path=exp_path,
             objects_group=objects_group,
             assignment_in_key=assignment_in_key,
             assignment_out_key=assignment_out_key,
             ws_key=ws_key, out_key=seg_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Resolving failed"
    return seg_key


def solve_jointly(path, exp_path, tmp_folder, target, max_jobs, n_threads):
    task = LiftedMulticutSegmentationWorkflow
    seg_key = 'volumes/segmentation/jointly'

    assignment_out_key = 'node_labels/jointly'

    input_key = 'volumes/affinities'
    ws_key = 'volumes/segmentation/watershed'
    mask_key = 'volumes/masks/minfilter/s5'

    os.makedirs(tmp_folder, exist_ok=True)
    # make log for costs in order to skip all the problem computation
    copyfile('./tmp_mc/probs_to_costs.log',
             os.path.join(tmp_folder, 'probs_to_costs.log'))

    config_dir = './configs'
    configs = task.get_config()
    conf_names = ['solve_lifted_subproblems',
                  'solve_lifted_global',
                  'reduce_lifted_problem']
    for name in conf_names:
        conf = configs[name]
        conf.update({'threads_per_job': n_threads, 'mem_limit': 128,
                     'time_limit': 150, 'time_limit_solver': 90 * 60 * 60,
                     'agglomerator': 'greedy-additive'})
        with open(os.path.join(config_dir, '%s.config' % name), 'w') as f:
            json.dump(conf, f)

    n_scales = 1
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target, max_jobs_multicut=8,
             input_path=path, input_key=input_key,
             mask_path=path, mask_key=mask_key,
             ws_path=path, ws_key=ws_key,
             problem_path=exp_path,
             node_labels_key=assignment_out_key,
             output_path=exp_path, output_key=seg_key,
             lifted_labels_path='', lifted_labels_key='',
             lifted_prefix='resolving',
             n_scales=n_scales, skip_ws=True)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Resolving failed"
    return seg_key


def parse_runtimes(tmp_folder):
    return None, None


def roc_point(recall, precision, target, max_jobs, n):
    path = '/g/kreshuk/data/FIB25/data.n5'
    gt_key1 = 'volumes/segmentation/groundtruth/cropped_google'
    gt_key2 = 'volumes/segmentation/groundtruth/cropped_full'

    tmp_folder = './tmp_roc/tmp_%s_%s' % (str(recall), str(precision))
    exp_path = os.path.join(tmp_folder, 'data.n5')
    config_dir = './configs'

    tmp_eval_sep = os.path.join(tmp_folder, 'tmp_eval_sep')
    tmp_eval_joint = os.path.join(tmp_folder, 'tmp_eval_joint')

    n_threads = 32
    result1 = []
    result2 = []
    for ii in range(n):
        # 1.) make the tmp and data folder
        os.makedirs(exp_path, exist_ok=True)

        # 2.) make the lifted problem
        lifted_problem(tmp_folder, config_dir, target, n_threads,
                       path, exp_path, recall, precision)

        # 3.) solve separate lifted problem
        key_sep = solve_separately(path, exp_path,
                                   tmp_folder, target, max_jobs, n_threads)

        # 4.) solve joint lifted problem
        key_joint = solve_jointly(path, exp_path,
                                  tmp_folder, target, max_jobs, n_threads)
        # rt_sep, rt_joint = parse_runtimes(tmp_folder)

        # 5.) copy and crop the two segmentations
        key_joint_c = 'volumes/segmentation/jointly_cropped'
        copy_and_crop_seg(exp_path, key_joint,
                          exp_path, key_joint_c,
                          config_dir,
                          os.path.join(tmp_folder, 'tmp_crop_joint'),
                          target='local', max_threads=n_threads)
        key_sep_c = 'volumes/segmentation/separately_cropped'
        copy_and_crop_seg(exp_path, key_sep,
                          exp_path, key_sep_c,
                          config_dir,
                          os.path.join(tmp_folder, 'tmp_crop_sep'),
                          target='local', max_threads=n_threads)

        # 6.) a) evaluate google groundtruth
        res_sep = evaluate_fib(exp_path, key_sep_c,
                               path, gt_key1,
                               tmp_eval_sep, config_dir,
                               target, max_jobs, n_threads)
        res_joint = evaluate_fib(exp_path, key_joint_c,
                                 path, gt_key1,
                                 tmp_eval_joint, config_dir,
                                 target, max_jobs, n_threads)
        result1.append({'jointly': res_joint, 'separately': res_sep})

        # 6.) b) evaluate google groundtruth
        res_sep = evaluate_fib(exp_path, key_sep_c,
                               path, gt_key2,
                               tmp_eval_sep, config_dir,
                               target, max_jobs, n_threads)
        res_joint = evaluate_fib(exp_path, key_joint_c,
                                 path, gt_key2,
                                 tmp_eval_joint, config_dir,
                                 target, max_jobs, n_threads)
        result2.append({'jointly': res_joint, 'separately': res_sep})

        # clean up
        print("Removing tmp")
        rmtree(tmp_folder)
    return result1, result2


def roc():
    out_path1 = 'roc_results_googlegt.json'
    out_path2 = 'roc_results_fullgt.json'

    if os.path.exists(out_path1):
        with open(out_path1, 'r') as f:
            results1 = json.load(f)
    else:
        results1 = {}

    if os.path.exists(out_path2):
        with open(out_path2, 'r') as f:
            results2 = json.load(f)
    else:
        results2 = {}

    # TODO full evaluation range
    rec_values = np.linspace(0.5, 1., 11)
    prec_values = np.linspace(0.5, 1., 11)

    target = 'slurm'
    max_jobs = 250
    # target = 'local'
    # max_jobs = 48

    for rec in rec_values:
        for prec in prec_values:

            exp_key = str(rec) + '_' + str(prec)

            # Don't need to evaluate for perfect recall / precision
            if rec == prec == 1.:
                continue

            # if exp_key in results1:
            #     continue

            print("Evaluationg rec", rec, "prec", prec)
            res1 = results1.get(exp_key, None)
            res2 = results2.get(exp_key, None)

            # TODO ideally, we want more statistics, n=5-10
            n = 4
            new_res1, new_res2 = roc_point(rec, prec, target, max_jobs, n=n)

            if res1 is None:
                results1[exp_key] = new_res1
            else:
                res1.extend(new_res1)
                results1[exp_key] = res1

            if res2 is None:
                results2[exp_key] = new_res2
            else:
                res2.extend(new_res2)
                results2[exp_key] = res2

            with open(out_path1, 'w') as f:
                json.dump(results1, f)
            with open(out_path2, 'w') as f:
                json.dump(results2, f)


if __name__ == '__main__':
    roc()
