#! /bin/python

import os
import sys
import json
from concurrent import futures

import luigi
import numpy as np
import nifty.distributed as ndist

import cluster_tools.utils.validation_utils as val
import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Validation measure Tasks
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
    path = os.path.join(input_path, overlap_key)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(ndist.deserializeOverlapChunk, path, [chunk_id])
                 for chunk_id in range(n_chunks)]
        results = [t.result()[0] for t in tasks]
    overlaps = {}
    for res in results:
        overlaps.update(res)

    # make contigency table objects, cf.
    # https://github.com/constantinpape/cluster_tools/blob/master/cluster_tools/utils/validation_utils.py#L9
    p_ids = np.array([[ida, idb] for ida, ovlp in overlaps.items()
                      for idb in ovlp.keys()])
    p_counts = np.array([ovlp_cnt for ovlp in overlaps.values()
                         for ovlp_cnt in ovlp.values()], dtype='uint64')
    ids_a = np.unique(p_ids[:, 0])
    ids_b = np.unique(p_ids[:, 1])

    sizes_a = np.array([np.sum(p_counts[p_ids[:, 0] == id_a])
                        for id_a in ids_a])
    sizes_b = np.array([np.sum(p_counts[p_ids[:, 1] == id_b])
                        for id_b in ids_b])

    a_dict = dict(zip(ids_a, sizes_a))
    b_dict = dict(zip(ids_b, sizes_b))

    # compute the total number of points
    n_points = np.sum(sizes_a)
    # consistency check
    assert n_points == np.sum(sizes_b) == np.sum(p_counts)

    # compute and save voi and rand measures
    vis, vim = val.compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points, True)
    ari, ri = val.compute_rand_scores(a_dict, b_dict, p_counts, n_points)

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
