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
    skeleton_format = luigi.Parameter(default='volume')
    dependency = luigi.TaskParameter(default=DummyTask())

    # TODO if format is not 'volume', we could also support more than 1 job
    formats = ('volume',)  # TODO support swc, n5-varlen ...

    def requires(self):
        return self.dependency

    def run_impl(self):
        assert self.skeleton_format in self.formats, self.skeleton_format
        # get the global config and init configs
        shebang, block_shape, _, _ = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape

        # load the skeletonize config
        task_config = self.get_task_config()

        # TODO need to adapt this once we support different output formats
        # require output dataset
        chunks = tuple(min(bs // 2, sh) for bs, sh in zip(block_shape, shape))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key,
                            'skeleton_format': self.skeleton_format})

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


def skeletonize_segment(seg_mask):
    # skimage transforms to uint8 and assigns maxval to skelpoints
    return np.where(skeletonize_3d(seg_mask) == 255)


def _skeletonize_to_volume(seg, ids, output_path, output_key, n_threads):
    with futures.ProcessPoolExecutor(n_threads) as pp:
        tasks = [pp.submit(skeletonize_segment, seg == seg_id) for seg_id in ids]
        results = {seg_id: t.result() for seg_id, t in zip(ids, tasks)}

    # allocate output
    skel_vol = np.zeros_like(seg, dtype='uint64')

    def write_res(seg_id):
        res = results[seg_id]
        skel_vol[res] = seg_id

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(write_res, seg_id) for seg_id in ids]
        [t.result() for t in tasks]

    # write the output
    with vu.file_reader(output_path) as f_out:
        ds_out = f_out[output_key]
        ds_out.n_threads = n_threads
        ds_out[:] = skel_vol


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
    skeleton_format = config['skeleton_format']

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
    fu.log("writing output to format %s" % skeleton_format)
    if skeleton_format == 'volume':
        _skeletonize_to_volume(seg, ids, output_path, output_key, n_threads)
    elif skeleton_format == 'swc':
        # TODO implement
        _skeletonize_to_swc(seg, ids, output_path, output_key, n_threads)
    else:
        raise RuntimeError("Format %s not supported" % skeleton_format)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    skeletonize(job_id, path)
