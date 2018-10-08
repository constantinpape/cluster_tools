#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import vigra
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class FindUniquesBase(luigi.Task):
    """ FindUniques base class
    """

    task_name = 'find_uniques'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    dependency = luigi.TaskParameter()
    return_counts = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)

        # we don't need any additional config besides the paths
        config = {"input_path": self.input_path, "input_key": self.input_key,
                  "block_shape": block_shape, "tmp_folder": self.tmp_folder,
                  "return_counts": self.return_counts}
        self._write_log('scheduling %i blocks to be processed' % len(block_list))

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class FindUniquesLocal(FindUniquesBase, LocalTask):
    """
    FindUniques on local machine
    """
    pass


class FindUniquesSlurm(FindUniquesBase, SlurmTask):
    """
    FindUniques on slurm cluster
    """
    pass


class FindUniquesLSF(FindUniquesBase, LSFTask):
    """
    FindUniques on lsf cluster
    """
    pass


#
# Implementation
#


def uniques_in_block(block_id, blocking, ds, return_counts):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    labels = ds[bb]

    if return_counts:
        uniques, counts = np.unique(labels, return_counts=True)
        fu.log_block_success(block_id)
        return uniques, counts
    else:
        uniques = nt.unique(labels)
        fu.log_block_success(block_id)
        return uniques


def find_uniques(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    block_list = config['block_list']
    block_shape = config['block_shape']
    tmp_folder = config['tmp_folder']
    return_counts = config['return_counts']

    # open the input file
    with vu.file_reader(input_path, 'r') as f:
        ds = f[input_key]
        shape = ds.shape
        blocking = nt.blocking(roiBegin=[0, 0, 0],
                               roiEnd=list(shape),
                               blockShape=list(block_shape))

        # find uniques for all blocks
        uniques = [uniques_in_block(block_id, blocking, ds, return_counts)
                   for block_id in block_list]

    if return_counts:
        unique_values = nt.unique(np.concatenate([un[0] for un in uniques]))
        counts = np.zeros(int(unique_values[-1] + 1), dtype='uint64')
        for uniques_block, counts_block in uniques:
            counts[uniques_block] += counts_block.astype('uint64')
        counts = counts[counts != 0]
        assert len(counts) == len(unique_values)

        count_path = os.path.join(tmp_folder, 'counts_job_%i.npy' % job_id)
        np.save(count_path, count_values)

    else:
        unique_values = nt.unique(np.concatenate(uniques))

    # save the uniques for this job
    save_path = os.path.join(tmp_folder, 'find_uniques_job_%i.npy' % job_id)
    fu.log("saving results to %s" % save_path)
    np.save(save_path, unique_values)
    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    find_uniques(job_id, path)
