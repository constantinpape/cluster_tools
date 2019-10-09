#! /usr/bin/python

import os
import sys
import json
from concurrent import futures

import luigi
import numpy as np

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Find Labeling Tasks
#

class MergeUniquesBase(luigi.Task):
    """ MergeUniques base class
    """

    task_name = 'merge_uniques'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    # where to save the assignments
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        shape = vu.get_shape(self.input_path, self.input_key)
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        config = self.get_task_config()
        config.update({'shape': shape,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'tmp_folder': self.tmp_folder, 'n_jobs': n_jobs})

        # we only have a single job to find the labeling
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(1)


class MergeUniquesLocal(MergeUniquesBase, LocalTask):
    """
    MergeUniques on local machine
    """
    pass


class MergeUniquesSlurm(MergeUniquesBase, SlurmTask):
    """
    MergeUniques on slurm cluster
    """
    pass


class MergeUniquesLSF(MergeUniquesBase, LSFTask):
    """
    MergeUniques on lsf cluster
    """
    pass


def merge_uniques(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    n_jobs = config['n_jobs']
    tmp_folder = config['tmp_folder']
    n_threads = config['threads_per_job']
    output_path = config['output_path']
    output_key = config['output_key']

    def _read_input(job_id):
        return np.load(os.path.join(tmp_folder, 'find_uniques_job_%i.npy' % job_id))

    fu.log("read uniques")
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_read_input, job_id) for job_id in range(n_jobs)]
        uniques = np.concatenate([t.result() for t in tasks])

    fu.log("compute uniques")
    uniques = np.unique(uniques)
    fu.log("found %i unique values" % len(uniques))

    fu.log("saving results to %s/%s" % (output_path, output_key))
    with vu.file_reader(output_path) as f:
        chunk_size = min(int(1e6), len(uniques))
        chunks = (chunk_size,)
        ds = f.create_dataset(output_key, shape=uniques.shape, dtype='uint64',
                              compression='gzip', chunks=chunks)
        ds.n_threads = n_threads
        ds[:] = uniques

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_uniques(job_id, path)
