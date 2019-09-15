#! /bin/python

import os
import sys
import json
import luigi
import z5py

import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Validation measure Tasks
#

class CopyAndCropBase(luigi.Task):
    """ CopyAndCrop base class
    """

    task_name = 'copy_and_crop'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    roi_start = luigi.ListParameter()
    roi_size = luigi.ListParameter()

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'roi_start': self.roi_start, 'roi_size': self.roi_size})

        n_jobs = 1
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class CopyAndCropLocal(CopyAndCropBase, LocalTask):
    """ CopyAndCrop on local machine
    """
    pass


class CopyAndCropSlurm(CopyAndCropBase, SlurmTask):
    """ CopyAndCrop on slurm cluster
    """
    pass


class CopyAndCropLSF(CopyAndCropBase, LSFTask):
    """ CopyAndCrop on lsf cluster
    """
    pass


#
# Implementation
#


def copy_and_crop(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    roi_start = config['roi_start']
    roi_size = config['roi_size']
    bb = tuple(slice(rs, rs + size) for rs, size in zip(roi_start, roi_size))
    max_threads = config.get('threads_per_job', 1)

    ds_in = z5py.File(input_path)[input_key]
    ds_in.n_threads = max_threads
    seg = ds_in[bb]
    max_id = int(seg.max())

    f = z5py.File(output_path)
    ds_out = f.require_dataset(output_key, shape=seg.shape, chunks=ds_in.chunks,
                               compression='gzip', dtype='uint64')
    ds_out.n_threads = max_threads
    ds_out[:] = seg
    ds_out.attrs['maxId'] = max_id

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    copy_and_crop(job_id, path)
