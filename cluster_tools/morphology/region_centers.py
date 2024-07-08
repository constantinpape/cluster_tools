#! /bin/python

# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import os
import sys
import json

import numpy as np
import luigi
import nifty.tools as nt
from scipy.ndimage import distance_transform_edt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Region Center Tasks
#

class RegionCentersBase(luigi.Task):
    """ RegionCenters base class
    """

    task_name = 'region_centers'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    morphology_path = luigi.Parameter()
    morphology_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    ignore_label = luigi.Parameter(default=None)
    resolution = luigi.ListParameter(default=[1, 1, 1])
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
        number_of_labels = int(vu.file_reader(self.input_path,
                                              'r')[self.input_key].attrs['maxId']) + 1
        # TODO should be a parameter
        id_chunks = 2000
        block_list = vu.blocks_in_volume([number_of_labels], [id_chunks])

        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=(number_of_labels, 3),
                              chunks=(id_chunks, 3), dtype='float32', compression='gzip')

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'morphology_path': self.morphology_path, 'morphology_key': self.morphology_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'ignore_label': self.ignore_label, 'resolution': self.resolution,
                       'id_chunks': id_chunks})

        # prime and run the jobs
        n_jobs = min(self.max_jobs, len(block_list))
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class RegionCentersLocal(RegionCentersBase, LocalTask):
    """ RegionCenters on local machine
    """
    pass


class RegionCentersSlurm(RegionCentersBase, SlurmTask):
    """ RegionCenters on slurm cluster
    """
    pass


class RegionCentersLSF(RegionCentersBase, LSFTask):
    """ RegionCenters on lsf cluster
    """
    pass


#
# Implementation
#


@threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
def region_centers_for_label_range(ds_in, ds_out, bb_start, bb_stop,
                                   label_begin, label_end,
                                   ignore_label, resolution):

    n_labels = label_end - label_begin
    centers = np.zeros((n_labels, 3), dtype='float32')
    for label_id in range(label_begin, label_end):
        if ignore_label == label_id:
            continue

        bb = tuple(slice(start, stop) for start, stop in
                   zip(bb_start[label_id], bb_stop[label_id]))
        obj = ds_in[bb] == label_id

        # can't do anything if the object is empty
        if obj.sum() == 0:
            continue

        dist = distance_transform_edt(obj, sampling=resolution)
        center = np.argmax(dist)
        center = np.unravel_index([center], obj.shape)

        offset = tuple(b.start for b in bb)
        center = [ce[0] + off for ce, off in zip(center, offset)]

        centers[label_id - label_begin] = center

    ds_out[label_begin:label_end] = centers


def region_centers(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    morphology_path = config['morphology_path']
    morphology_key = config['morphology_key']
    output_path = config['output_path']
    output_key = config['output_key']

    ignore_label = config['ignore_label']
    resolution = config['resolution']

    block_list = config['block_list']
    id_chunks = config['id_chunks']
    number_of_labels = int(vu.file_reader(input_path, 'r')[input_key].attrs['maxId']) + 1
    blocking = nt.blocking([0], [number_of_labels], [id_chunks])

    # load the morphology and get the relevant cols
    with vu.file_reader(morphology_path) as f:
        ds = f[morphology_key]
        morphology = ds[:]
    bb_start = morphology[:, 5:8].astype('uint64')
    bb_stop = morphology[:, 8:11].astype('uint64')

    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        for block_id in block_list:
            block = blocking.getBlock(block_id)
            label_begin = block.begin[0]
            label_end = block.end[0]
            region_centers_for_label_range(ds_in, ds_out, bb_start, bb_stop,
                                           label_begin, label_end,
                                           ignore_label, resolution)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    region_centers(job_id, path)
