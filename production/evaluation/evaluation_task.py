#! /usr/bin/python

import os
import argparse
import subprocess
import json

import luigi
import z5py
from cremi_tools.skeletons import build_skeleton_metrics

from ..util import DummyTask


class SkeletonEvaluationTask(luigi.Task):
    path = luigi.Parameter()
    seg_key = luigi.Parameter()
    skeleton_keys = luigi.ListParameter()
    n_threads = luigi.IntParameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter(default=DummyTask())
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    # TODO enable ROIs
    def run(self):

        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(self.tmp_folder, 'evaluation_task.py')
        util.copy_and_replace(os.path.join(file_dir, 'evaluation_task.py'),
                              script_path)

        # check that inputs exist
        f = z5py.File(self.path)
        assert self.seg_key in f
        assert all(skeleton_key in f for skeleton_key in self.skeleton_keys)

        # prepare the job
        command = '%s %s %s --skeleton_keys %s --n_threads %i --tmp_fold er%s %s' % (script_path,
                                                                                     self.path,
                                                                                     self.seg_key,
                                                                                     ' '.join(self.skeleton_keys),
                                                                                     self.n_threads,
                                                                                     self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_skeleton_eval.log')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_skeleton_eval.err')
        bsub_command = 'bsub -n %i -J compute_skeleton_eval -We %i -o %s -e %s \'%s\'' % (self.n_threads,
                                                                                          self.time_estimate,
                                                                                          log_file, err_file, command)

        # submit the job
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)
            util.wait_for_jobs('papec')

        # load and check the output
        out_path = self.output().path
        try:
            with open(out_path) as f:
                evaluation = json.load(f)
            for key, eva in evaluation.items:
                print("Skeleton evaliation for %s:" % key)
                print("Correct:     ", eva['correct'])
                print("Split:       ", eva['split'])
                print("Merge:       ", eva['merge'])
                print("Merge Points:", eva['n_merges'])
        except Exception:
            raise RuntimeError("SkeletonEvaluationTask failed")

    def output(self):
        res_path = os.path.join(self.tmp_folder, 'skeleton_eval_res.json')
        return luigi.LocalTarget(res_path)


# TODO save timing
def compute_skeleton_evaluation(path, seg_key, skeleton_keys, n_threads, tmp_folder):

    label_file = os.path.join(path, seg_key)

    results = {}
    for skel_key in skeleton_keys:
        skeleton_file = os.path.join(path, skel_key)
        metrics = build_skeleton_metrics(label_file, skeleton_file, n_threads)
        correct, split, merge, n_merges = metrics.computeGoogleScore(n_threads)
        res = {'correct': correct, 'split': split, 'merge': merge, 'n_merges': n_merges}
        results[skel_key] = res

    res_path = os.path.join(tmp_folder, 'skeleton_eval_res.json')
    with open(res_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('seg_key', type=str)
    parser.add_argument('--skeleton_keys', type=str, nargs='+')
    parser.add_argument('--n_threads', type=int)
    parser.add_argument('--tmp_folder', type=str)

    args = parser.parse_args()
    compute_skeleton_evaluation(args.path, args.seg_key,
                                list(args.skeleton_keys), args.n_threads,
                                args.tmp_folder)
