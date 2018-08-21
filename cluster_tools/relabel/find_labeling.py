#! /usr/bin/python

import os
import sys
import json
import pickle

import luigi
import numpy as np
import vigra
import nifty.tools as nt

import cluster_tools.volume_util as vu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.functional_api import log_job_success, log_block_success, log, load_global_config


#
# Find Labeling Tasks
#

class FindLabelingBase(luigi.Task):
    """ FindLabeling base class
    """

    task_name = 'find_labeling'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = load_global_config(self.global_config_path)
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        config = {'input_path': self.input_path, 'input_key': self.input_key,
                  'tmp_folder': self.tmp_folder, 'n_jobs': n_jobs}

        # we only have a single job to find the labeling
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class FindLabelingLocal(FindLabelingBase, LocalTask):
    """
    FindLabeling on local machine
    """
    pass


class FindLabelingSlurm(FindLabelingBase, SlurmTask):
    """
    FindLabeling on slurm cluster
    """
    pass


class FindLabelingLSF(FindLabelingBase, LSFTask):
    """
    FindLabeling on lsf cluster
    """
    pass



def find_labeling(job_id, config_path):

    log("start processing job %i" % job_id)
    log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    n_jobs = config['n_jobs']
    tmp_folder = config['tmp_folder']
    input_path = config['input_path']
    input_key = config['input_key']

    # TODO this could be parallelized
    uniques = np.concatenate([np.load(os.path.join(tmp_folder, 'uniques_job_%i.npy' % job_id))
                              for job_id in range(n_jobs)])
    uniques = nt.unique(uniques)
    _, max_id, mapping = vigra.analysis.relabelConsecutive(uniques,
                                                           keep_zeros=True,
                                                           start_label=1)

    with vu.file_reader(input_path) as f:
        f[input_key].attrs['maxId'] = max_id

    save_path = os.path.join(tmp_folder, 'relabeling.pkl')
    log("saving results to %s" % save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(mapping, f)
    # log success
    log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    find_labeling(job_id, path)
