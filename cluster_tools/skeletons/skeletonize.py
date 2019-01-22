#! /bin/python

import os
import sys
import json
from concurrent import futures

import numpy as np
import luigi
import nifty.tools as nt
from skimage.morphology import skeletonize_3d

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.skeleton_utils as su
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
    number_of_labels = luigi.IntParameter()
    skeleton_format = luigi.Parameter(default='n5')
    dependency = luigi.TaskParameter(default=DummyTask())

    formats = ('volume', 'swc', 'n5')  # TODO support csv

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'resolution': None, 'size_filter': 10,
                       'chunk_len': 1000})
        return config

    def requires(self):
        return self.dependency

    def _prepare_format_volume(self, block_shape):
        assert self.max_jobs == 1, "Output-format 'volume' only supported with a single job"

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape

        # prepare output dataset
        chunks = tuple(min(bs // 2, sh) for bs, sh in zip(block_shape, shape))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')
        # return n-jobs (=1) and block_list (=None)
        return 1, None

    def _prepare_format_swc(self, config):
        # make the output directory
        os.makedirs(os.path.join(self.output_path, self.output_key),
                    exist_ok=True)
        # make the blocking
        block_len = min(self.number_of_labels, config.get('chunk_len', 1000))
        block_list = vu.blocks_in_volume((self.number_of_labels,),
                                         (block_len,))
        n_jobs = min(len(block_list), self.max_jobs)
        # update the config
        config.update({'number_of_labels': self.number_of_labels,
                       'block_len': block_len})
        return n_jobs, block_list

    def _prepare_format_n5(self, config):
        # make the blocking
        block_len = min(self.number_of_labels, config.get('chunk_len', 1000))
        block_list = vu.blocks_in_volume((self.number_of_labels,),
                                         (block_len,))
        n_jobs = min(len(block_list), self.max_jobs)
        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=(self.number_of_labels,),
                              chunks=(1,), compression='gzip', dtype='uint64')
        # update the config
        config.update({'number_of_labels': self.number_of_labels,
                       'block_len': block_len})
        return n_jobs, block_list

    def run_impl(self):
        assert self.skeleton_format in self.formats, self.skeleton_format
        # TODO support roi
        # get the global config and init configs
        shebang, block_shape, _, _ = self.global_config_values()
        self.init(shebang)

        # load the skeletonize config
        # update the config with input and output paths and keys
        task_config = self.get_task_config()
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key,
                            'skeleton_format': self.skeleton_format})

        if self.skeleton_format == 'volume':
            n_jobs, block_list = self._prepare_format_volume(block_shape)
        elif self.skeleton_format == 'swc':
            n_jobs, block_list = self._prepare_format_swc(task_config)
        elif self.skeleton_format == 'n5':
            n_jobs, block_list = self._prepare_format_n5(task_config)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, task_config)
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


#
# functions for 'volume' skeleton format
#

def _get_ids(seg, size_filter):
    if size_filter > 0:
        ids, counts = np.unique(seg, return_counts=True)
        ids = ids[counts > size_filter]
    else:
        ids = np.unique(seg)
    # if 0 in ids, discard it (ignore id)
    if ids[0] == 0:
        ids = ids[1:]
    return ids


def skeletonize_segment(seg_mask):
    # skimage transforms to uint8 and assigns maxval to skelpoints
    return np.where(skeletonize_3d(seg_mask) == 255)


def _skeletonize_to_volume(seg, output_path, output_key, config):

    size_filter = config.get('size_filter', 10)
    n_threads = config.get('threads_per_job', 1)
    ids = _get_ids(seg, size_filter)

    fu.log("computing skeletons for %i ids" % len(ids))
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


#
# functions for 'swc' skeleton format
#

def _skeletonize_id_to_swc(seg, node_id, size_filter, output_path, resolution):
    mask = seg == node_id
    if np.sum(mask) <= size_filter:
        return
    skel_vol = skeletonize_3d(mask)
    # write skeleton as swc
    su.write_swc(output_path, skel_vol, resolution)


# not parallelized for now
def _skeletonize_to_swc(seg, output_path, output_key, config):
    block_list = config['block_list']
    block_len = config['block_len']
    n_labels = config['number_of_labels']
    blocking = nt.blocking([0], [n_labels], [block_len])

    resolution = config.get('resolution', None)

    output_folder = os.path.join(output_path, output_key)
    size_filter = config.get('size_filter', 10)
    for block_id in block_list:
        fu.log("start processing block %i" % block_id)
        block = blocking.getBlock(block_id)
        id_begin, id_end = block.begin[0], block.end[0]
        # we don't compute the skeleton for id 0, which is reserved for the ignore label
        id_begin = 1 if id_begin == 0 else id_begin

        for node_id in range(id_begin, id_end):
            out_path = os.path.join(output_folder, '%i.swc' % node_id)
            _skeletonize_id_to_swc(seg, node_id, size_filter, out_path, resolution)
        fu.log_block_success(block_id)


#
# functions for 'n5' skeleton format
#

def _skeletonize_id_to_n5(seg, node_id, size_filter, ds):
    mask = seg == node_id
    if np.sum(mask) <= size_filter:
        return
    coords = np.where(mask)
    min_coords = [np.min(coord) for coord in coords]
    max_coords = [np.max(coord) for coord in coords]
    bb = tuple(slice(minc, maxc + 1)
               for minc, maxc in zip(min_coords, max_coords))
    skel_vol = skeletonize_3d(mask[bb])
    # write skeleton as swc
    su.write_n5(ds, node_id, skel_vol,
                coordinate_offset=min_coords)


# not parallelized for now
def _skeletonize_to_n5(seg, output_path, output_key, config):
    block_list = config['block_list']
    block_len = config['block_len']
    n_labels = config['number_of_labels']
    blocking = nt.blocking([0], [n_labels], [block_len])

    ds = vu.file_reader(output_path)[output_key]
    size_filter = config.get('size_filter', 10)
    for block_id in block_list:
        fu.log("start processing block %i" % block_id)
        block = blocking.getBlock(block_id)
        id_begin, id_end = block.begin[0], block.end[0]
        # we don't compute the skeleton for id 0, which is reserved for the ignore label
        id_begin = 1 if id_begin == 0 else id_begin

        for node_id in range(id_begin, id_end):
            _skeletonize_id_to_n5(seg, node_id, size_filter, ds)
        fu.log_block_success(block_id)

#
# main function
#

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

    fu.log("writing output in format %s" % skeleton_format)
    if skeleton_format == 'volume':
        _skeletonize_to_volume(seg, output_path, output_key, config)
    elif skeleton_format == 'swc':
        _skeletonize_to_swc(seg, output_path, output_key, config)
    elif skeleton_format == 'n5':
        _skeletonize_to_n5(seg, output_path, output_key, config)
    else:
        raise RuntimeError("Format %s not supported" % skeleton_format)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    skeletonize(job_id, path)
