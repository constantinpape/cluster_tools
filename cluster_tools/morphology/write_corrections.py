#! /bin/python

import os
import sys
import json

import luigi
import numpy as np

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class WriteCorrectionsBase(luigi.Task):
    """ WriteCorrections base class
    """

    task_name = 'write_corrections'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    morphology_path = luigi.Parameter()
    morphology_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'output_path': self.output_path,
                       'output_key': self.output_key,
                       'morphology_path': self.morphology_path,
                       'morphology_key': self.morphology_key,
                       'tmp_folder': self.tmp_folder,
                       'n_jobs': self.max_jobs})

        n_jobs = 1
        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class WriteCorrectionsLocal(WriteCorrectionsBase, LocalTask):
    """ WriteCorrections on local machine
    """
    pass


class WriteCorrectionsSlurm(WriteCorrectionsBase, SlurmTask):
    """ WriteCorrections on slurm cluster
    """
    pass


class WriteCorrectionsLSF(WriteCorrectionsBase, LSFTask):
    """ WriteCorrections on lsf cluster
    """
    pass


#
# Implementation
#


def _write_corrections_inplace(ids, centers, morphology_path, morphology_key):
    with vu.file_reader(morphology_path) as f:
        ds = f[morphology_key]
        label_ids = ds[:, 0].astype('uint64')
        com = ds[:, 2:5]

        correction_mask = np.in1d(label_ids, ids)
        com[correction_mask, :] = centers

        ds[:, 2:5] = com


def _write_corrections_new(ids, centers, morphology_path, morphology_key,
                           output_path, output_key):
    with vu.file_reader(morphology_path, 'r') as f:
        ds = f[morphology_key]
        label_ids = ds[:, 0].astype('uint64')
        com = ds[:, 2:5]

        correction_mask = np.in1d(label_ids, ids)
        com[correction_mask, :] = centers

    data = np.concatenate([label_ids[:, None].astype('float32'), com], axis=1)
    with vu.file_reader(output_path) as f:
        chunks = (min(len(data), 10000), data.shape[1])
        ds_out = f.require_dataset(output_key, shape=data.shape, chunks=chunks,
                                   compression='gzip', dtype='float32')
        ds_out[:] = data


def write_corrections(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    morphology_path = config['morphology_path']
    morphology_key = config['morphology_key']
    output_path = config['output_path']
    output_key = config['output_key']

    tmp_folder = config['tmp_folder']
    n_jobs = tmp_folder['n_jobs']

    # load all corrections from the correction jobs
    anchor_corrections = {}
    for job_id in n_jobs:
        tmp_path = os.path.join(tmp_folder, 'corrections_job%i.json' % job_id)

        # we don't bother to find the actual number of jobs, so some of these might not exist
        if not os.path.exists(tmp_path):
            continue

        with open(tmp_path, 'r') as f:
            corrections = json.load(f)
        # json cast's all keys to int, we need to go back to int
        corrections = {int(label_id): center for label_id, center in corrections.items()}
        anchor_corrections.update(corrections)

    fu.log("Correcting anchors for %i ids" % len(anchor_corrections))
    ids = np.array(list(anchor_corrections.keys()))
    centers = np.array([center for center in anchor_corrections.values()])
    assert len(ids) == len(centers)

    # write new corrections, in-place or to new file
    if output_path == '':
        _write_corrections_inplace(ids, centers, morphology_path, morphology_key)
    else:
        assert output_key != ''
        _write_corrections_new(ids, centers, morphology_path, morphology_key,
                               output_path, output_key)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    write_corrections(job_id, path)
