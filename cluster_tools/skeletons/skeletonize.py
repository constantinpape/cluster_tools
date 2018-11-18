#! /bin/python

import os
import sys
import json
from functools import partial
from concurrent import futures

import numpy as np
import luigi
from skimage.morphology import skeletonize_3d

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# skeletonize tasks
#


# TODO it would make more sense to output skeletons in swc format
# or some other format
class SkeletonizeBase(luigi.Task):
    """ Skeletonize base class
    """

    task_name = 'skeletonize'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape

        # load the skeletonize config
        task_config = self.get_task_config()

        # require output dataset
        chunks = (25, 256, 256)
        chunks = tuple(min(sh, ch) for sh, ch in zip(shape, chunks))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key})

        # prime and run the jobs
        n_jobs = 1
        self.prepare_jobs(n_jobs, None, task_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class SkeletonizeLocal(SkeletonizeBase, LocalTask):
    """
    skeletonize on local machine
    """
    pass


class SkeletonizeSlurm(SkeletonizeBase, SlurmTask):
    """
    skeletonize on slurm cluster
    """
    pass


class SkeletonizeLSF(SkeletonizeBase, LSFTask):
    """
    skeletonize on lsf cluster
    """
    pass


#
# Implementation
#


def skeletonize_multithreaded(seg, ids, n_threads):
    # allocate output
    skel_vol = np.zeros_like(seg, dtype='uint64')

    def skeletonize_id(seg_id):
        seg_mask = seg == seg_id
        # FIXME: does not lift the gil
        skel = skeletonize_3d(seg_mask)
        # skimage transforms to uint8 and assigns maxval to skelpoints
        skel_vol[skel == 255] = seg_id

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(skeletonize_id, seg_id)
                 for seg_id in ids]
        [t.result() for t in tasks]
    return skel_vol


def skeletonize_segment(seg_mask):
    # skimage transforms to uint8 and assigns maxval to skelpoints
    return np.where(skeletonize_3d(seg_mask) == 255)


def skeletonize_mp(seg, ids, n_threads):
    # allocate output
    skel_vol = np.zeros_like(seg, dtype='uint64')

    with futures.ProcessPoolExecutor(n_threads) as pp:
        tasks = [pp.submit(skeletonize_segment, seg == seg_id) for seg_id in ids]
        results = {seg_id: t.result() for seg_id, t in zip(ids, tasks)}

    def write_res(seg_id):
        res = results[seg_id]
        skel_vol[res] = seg_id

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(write_res, seg_id) for seg_id in ids]
        [t.result() for t in tasks]
    return skel_vol


def skeletonize(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']

    output_path = config['output_path']
    output_key = config['output_key']

    n_threads = config.get('threads_per_job', 1)

    # load input segmentation
    with vu.file_reader(input_path) as f_in:
        ds_in = f_in[input_key]
        ds_in.n_threads = n_threads
        seg = ds_in[:]

    # TODO size filtering ?
    # find unique ids in the segmentation
    ids = np.unique(seg)
    # if 0 in ids, discard it (ignore id)
    if ids[0] == 0:
        ids = ids[1:]

    fu.log("computing skeletons for %i ids" % len(ids))
    # FIXME this is too slow because skeletonize 3d does not lift gil
    # skel_vol = skeletonize_multi_threaded(seg, ids, n_threads)
    skel_vol = skeletonize_mp(seg, ids, n_threads)

    # write the output
    with vu.file_reader(output_path) as f_out:
        ds_out = f_out[output_key]
        ds_out.n_threads = n_threads
        ds_out[:] = skel_vol


    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    skeletonize(job_id, path)
