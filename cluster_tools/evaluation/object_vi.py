#! /bin/python

import os
import sys
import json

import numpy as np
import luigi
from elf.evaluation.variation_of_information import compute_object_vi_scores

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

from cluster_tools.evaluation.measures import contigency_table_from_overlaps, load_overlaps


#
# Validation measure tasks
#

class ObjectViBase(luigi.Task):
    """ ObjectVi base class
    """

    task_name = 'object_vi'
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


class ObjectViLocal(ObjectViBase, LocalTask):
    """ ObjectVi on local machine
    """
    pass


class ObjectViSlurm(ObjectViBase, SlurmTask):
    """ ObjectVi on slurm cluster
    """
    pass


class ObjectViLSF(ObjectViBase, LSFTask):
    """ ObjectVi on lsf cluster
    """
    pass


#
# Implementation
#

def overlaps_to_sizes(pairs, counts):
    sorted_ids = np.argsort(pairs)
    sizes = counts[sorted_ids]
    sorted_ids = pairs[sorted_ids]
    _, label_starts = np.unique(sorted_ids, return_index=True)
    label_ends = label_starts[1:].tolist() + [sizes.size]
    sizes = np.array([np.sum(sizes[lstart:lend])
                      for lstart, lend in zip(label_starts, label_ends)])
    return sizes


def object_vi(job_id, config_path):

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
    object_scores = compute_object_vi_scores(a_dict, b_dict, p_ids, p_counts, use_log2=True)

    # annoying json ...
    object_scores = {int(gt_id): score for gt_id, score in object_scores.items()}
    with open(output_path, 'w') as f:
        json.dump(object_scores, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    object_vi(job_id, path)
