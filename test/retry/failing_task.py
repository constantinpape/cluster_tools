#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import LocalTask


#
# Task that (deterministically) fails to test the retry mechanism
#

class FailingTaskBase(luigi.Task):
    """ FailingTask base class
    """

    task_name = "failing_task"
    src_file = os.path.abspath(__file__)

    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    shape = luigi.ListParameter()

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        config = self.get_task_config()

        # make outout and update the config
        shape = tuple(self.shape)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=tuple(block_shape), dtype="uint8")

        config.update({"output_path": self.output_path, "output_key": self.output_key,
                       "n_retries": self.n_retries, "block_shape": block_shape})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class FailingTaskLocal(FailingTaskBase, LocalTask):
    """FailingTask on local machine
    """
    pass


def _failing_block(block_id, blocking, ds, n_retries):
    # fail for odd block ids if we are in the first try
    if n_retries == 0 and block_id % 2 == 1:
        raise RuntimeError("Fail")
    bb = vu.block_to_bb(blocking.getBlock(block_id))
    ds[bb] = 1
    fu.log_block_success(block_id)


def failing_task(job_id, config_path):

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    output_path = config["output_path"]
    output_key = config["output_key"]
    block_shape = config["block_shape"]
    block_list = config["block_list"]
    n_retries = config["n_retries"]

    shape = vu.get_shape(output_path, output_key)
    blocking = nt.blocking(roiBegin=[0, 0, 0],
                           roiEnd=list(shape),
                           blockShape=list(block_shape))

    with vu.file_reader(output_path) as f:
        ds = f[output_key]
        for block_id in block_list:
            _failing_block(block_id, blocking, ds, n_retries)
    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    failing_task(job_id, path)
