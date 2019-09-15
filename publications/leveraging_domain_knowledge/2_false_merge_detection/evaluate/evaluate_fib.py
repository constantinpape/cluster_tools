import os
import luigi
import json
import z5py

from cluster_tools.evaluation import EvaluationWorkflow


def evaluate_fib(seg_path, seg_key,
                 gt_path, gt_key,
                 tmp_folder, config_dir,
                 target, max_jobs, max_threads):
    task = EvaluationWorkflow
    with z5py.File(gt_path) as f:
        shape1 = f[gt_key].shape

    with z5py.File(seg_path) as f:
        shape2 = f[seg_key].shape

    assert shape1 == shape2, "%s, %s" % (str(shape1), str(shape2))

    global_conf = task.get_config()['global']
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    global_conf.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    out_path = os.path.join(tmp_folder, 'eval_res.json')
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             seg_path=seg_path, seg_key=seg_key,
             gt_path=gt_path, gt_key=gt_key,
             output_path=out_path, ignore_label=True)
    ret = luigi.build([t], local_scheduler=True)
    assert ret

    with open(out_path) as f:
        res = json.load(f)
    return res
