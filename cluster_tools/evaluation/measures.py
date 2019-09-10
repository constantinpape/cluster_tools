#! /bin/python

import os
import sys
import json
from concurrent import futures

import numpy as np
import luigi
import nifty.distributed as ndist
from elf.evaluation.rand_index import compute_rand_scores
from elf.evaluation.variation_of_information import compute_vi_scores

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Validation measure tasks
#

class MeasuresBase(luigi.Task):
    """ Measures base class
    """

    task_name = 'measures'
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


class MeasuresLocal(MeasuresBase, LocalTask):
    """ Measures on local machine
    """
    pass


class MeasuresSlurm(MeasuresBase, SlurmTask):
    """ Measures on slurm cluster
    """
    pass


class MeasuresLSF(MeasuresBase, LSFTask):
    """ Measures on lsf cluster
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


def contigency_table_from_overlaps(overlaps):
    # make contigency table objects, cf.
    # https://github.com/constantinpape/elf/blob/master/elf/evaluation/util.py#L22
    p_ids = np.array([[ida, idb] for idb, ovlp in overlaps.items()
                      for ida in ovlp.keys()])
    p_counts = np.array([ovlp_cnt for ovlp in overlaps.values()
                         for ovlp_cnt in ovlp.values()], dtype='float64')

    pairs_a = p_ids[:, 0]
    ids_a = np.unique(pairs_a)
    pairs_b = p_ids[:, 1]
    ids_b = np.unique(pairs_b)

    # get the sizes from the overlaps
    sizes_a = overlaps_to_sizes(pairs_a, p_counts)
    sizes_b = overlaps_to_sizes(pairs_b, p_counts)

    a_dict = dict(zip(ids_a, sizes_a))
    b_dict = dict(zip(ids_b, sizes_b))

    # compute the total number of points
    n_points = np.sum(sizes_a)
    # consistency check
    assert n_points == np.sum(sizes_b) == np.sum(p_counts)

    return a_dict, b_dict, p_ids, p_counts, n_points


def load_overlaps(path, key, n_chunks, n_threads):
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(ndist.deserializeOverlapChunk, path, key, [chunk_id])
                 for chunk_id in range(n_chunks)]
        results = [t.result()[0] for t in tasks]
    overlaps = {}
    for res in results:
        overlaps.update(res)
    return overlaps


def measures(job_id, config_path):

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
    a_dict, b_dict, p_ids, p_counts, n_points = contigency_table_from_overlaps(overlaps)

    # compute and save voi and rand measures
    vis, vim = compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points, True)
    ari, ri = compute_rand_scores(a_dict, b_dict, p_counts, n_points)

    results = {'vi-split': vis, 'vi-merge': vim,
               'adapted-rand-error': ari, 'rand-index': ri}
    with open(output_path, 'w') as f:
        json.dump(results, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    measures(job_id, path)
