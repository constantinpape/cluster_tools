#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
from scipy.optimize import linear_sum_assignment
from elf.evaluation.matching import intersection_over_union

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

from cluster_tools.evaluation.measures import contigency_table_from_overlaps, load_overlaps


#
# Validation measure tasks
#

class ObjectIouBase(luigi.Task):
    """ ObjectIou base class
    """

    task_name = 'object_iou'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    overlap_key = luigi.Parameter()
    output_path = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        config.update({'input_path': self.input_path, 'overlap_key': self.overlap_key,
                       'output_path': self.output_path})

        n_jobs = 1
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class ObjectIouLocal(ObjectIouBase, LocalTask):
    """ ObjectIou on local machine
    """
    pass


class ObjectIouSlurm(ObjectIouBase, SlurmTask):
    """ ObjectIou on slurm cluster
    """
    pass


class ObjectIouLSF(ObjectIouBase, LSFTask):
    """ ObjectIou on lsf cluster
    """
    pass


#
# Implementation
#


def compute_ious(p_ids, p_counts):
    # compute the label overlaps
    p_ids = p_ids.astype('uint64')
    max_a, max_b = int(p_ids[:, 0].max()), int(p_ids[:, 1].max())
    overlap = np.zeros((max_a + 1, max_b + 1), dtype='uint64')
    index = (p_ids[:, 0], p_ids[:, 1])
    overlap[index] = p_counts

    # compute the ious
    scores = intersection_over_union(overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    n_pred, n_true = scores.shape
    n_matched = min(n_true, n_pred)

    threshold = 0.5
    if not (scores > threshold).any():
        return dict(zip(range(n_pred), n_pred * [0.]))

    costs = -(scores >= threshold).astype(float) - scores / (2*n_matched)
    pred_ind, true_ind = linear_sum_assignment(costs)
    assert n_matched == len(true_ind) == len(pred_ind)
    scores = scores[pred_ind, true_ind]

    matched_ids = p_ids[pred_ind].tolist()
    scores = dict(zip(matched_ids, scores.tolist()))

    # set scores for non-matched ids to zero
    missing_ids = list(set(range(max_a + 1)) - set(matched_ids))
    scores.update({mid: 0. for mid in missing_ids})

    return scores


def object_iou(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    overlap_key = config['overlap_key']

    output_path = config['output_path']
    n_threads = config.get('threads_per_job', 1)

    f = vu.file_reader(input_path, 'r')

    # load overlaps in parallel and merge them
    n_chunks = f[overlap_key].number_of_chunks
    overlaps = load_overlaps(input_path, overlap_key, n_chunks, n_threads)

    a_dict, b_dict, p_ids, p_counts, _ = contigency_table_from_overlaps(overlaps)
    object_scores = compute_ious(p_ids, p_counts)

    # annoying json ...
    object_scores = {int(gt_id): score for gt_id, score in object_scores.items()}
    with open(output_path, 'w') as f:
        json.dump(object_scores, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    object_iou(job_id, path)
