#! /bin/python

import os
import sys
import json
import pickle
from concurrent import futures

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Write Tasks
#

class WriteBase(luigi.Task):
    """
    Write node assignments for all blocks
    """
    task_name = 'write'
    src_file = os.path.abspath(__file__)

    # path adn key to input and output datasets
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # path to the node assignments
    # the key is optional, because the assignment can either be a
    # dense assignment table stored as n5 dataset
    # or a sparse table stored as pickled python map
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter(default=None)
    # the task we depend on
    dependency = luigi.TaskParameter(default=DummyTask())
    # we may have different write tasks,
    # so we need an identifier to keep them apart
    identifier = luigi.Parameter()
    offset_path = luigi.Parameter(default='')

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'chunks': None, 'allow_empty_assignments': False})
        return config

    def clean_up_for_retry(self, block_list, prefix):
        super().clean_up_for_retry(block_list, prefix)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and chunks
        shape = vu.get_shape(self.input_path, self.input_key)

        config = self.get_task_config()
        chunks = config.pop('chunks', None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)
            chunks = tuple(min(ch, sh) for ch, sh in zip(chunks, shape))

        # require output dataset
        with vu.file_reader(self.output_path) as f:
            if self.output_key in f:
                chunks = f[self.output_key].chunks
            assert all(bs % ch == 0 for bs, ch in zip(block_shape, chunks)), "%s, %s" % (str(block_shape),
                                                                                         str(chunks))
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # check if input and output datasets are identical
        in_place = (self.input_path == self.output_path) and (self.input_key == self.output_key)

        if self.assignment_key is None:
            assert os.path.splitext(self.assignment_path)[-1] == '.pkl',\
                "Assignments need to be pickled map if no key is given"

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key, 'block_shape': block_shape,
                       'assignment_path': self.assignment_path, 'assignment_key': self.assignment_key})
        if self.offset_path != '':
            config.update({'offset_path': self.offset_path})
        # we only add output path and key if we do not write in place
        if not in_place:
            config.update({'output_path': self.output_path, 'output_key': self.output_key})

        # get block list and jobs
        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list, self.identifier)
        self._write_log('scheduling %i blocks to be processed' % len(block_list))

        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config, self.identifier)
        self.submit_jobs(n_jobs, self.identifier)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.identifier)
        self.check_jobs(n_jobs, self.identifier)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, '%s_%s.log' % (self.task_name,
                                                                              self.identifier)))


class WriteLocal(WriteBase, LocalTask):
    """ Write on local machine
    """
    pass


class WriteSlurm(WriteBase, SlurmTask):
    """ Write on slurm cluster
    """
    pass


class WriteLSF(WriteBase, LSFTask):
    """ Write on lsf cluster
    """
    pass


#
# Implementation
#


def _apply_node_labels(seg, node_labels, allow_empty_assignments):
    # choose the appropriate mapping:
    # - 1d np.array -> just apply it
    # - 2d np.array -> extract the local dict and apply
    # - dict -> extract the local dict and apply
    apply_array = False if isinstance(node_labels, dict) else (True if node_labels.ndim == 1 else False)
    if apply_array:
        seg = nt.take(node_labels, seg)
    else:
        # this copys the dict and hence is extremely RAM hungry
        # so we make the dict as small as possible
        this_labels = np.unique(seg)
        if isinstance(node_labels, dict):
            # do we allow for assignments that are not in the assignment table?
            if allow_empty_assignments:
                this_assignment = {label: node_labels.get(label, label) for label in this_labels}
            else:
                this_assignment = {label: node_labels[label] for label in this_labels}
        else:
            this_assignment = node_labels[:, 1][np.in1d(node_labels[:, 0], this_labels)]
            this_assignment = {label: this_assignment[ii] for ii, label in enumerate(this_labels)}
        seg = nt.takeDict(this_assignment, seg)
    return seg


def _write_block_with_offsets(ds_in, ds_out, blocking, block_id,
                              node_labels, offsets, allow_empty_assignments):
    fu.log("start processing block %i" % block_id)
    off = offsets[block_id]
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    seg = ds_in[bb]

    # check if this block is empty and don't write if it is
    mask = seg != 0
    if np.sum(mask) == 0:
        fu.log_block_success(block_id)
        return

    seg[mask] += off
    seg = _apply_node_labels(seg, node_labels, allow_empty_assignments)
    ds_out[bb] = seg
    fu.log_block_success(block_id)


def _write_with_offsets(ds_in, ds_out, blocking, block_list,
                        n_threads, node_labels, offset_path, allow_empty_assignments):

    fu.log("loading offsets from %s" % offset_path)
    with open(offset_path) as f:
        offset_config = json.load(f)
        offsets = offset_config['offsets']
        empty_blocks = offset_config['empty_blocks']

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_write_block_with_offsets, ds_in, ds_out,
                           blocking, block_id, node_labels, offsets, allow_empty_assignments)
                 for block_id in block_list if block_id not in empty_blocks]
        [t.result() for t in tasks]


def _write_block(ds_in, ds_out, blocking, block_id, node_labels, allow_empty_assignments):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    seg = ds_in[bb]
    # check if this block is empty and don't write if it is
    if np.sum(seg != 0) == 0:
        fu.log_block_success(block_id)
        return

    seg = _apply_node_labels(seg, node_labels, allow_empty_assignments)
    ds_out[bb] = seg
    fu.log_block_success(block_id)


def _write(ds_in, ds_out, blocking, block_list,
           n_threads, node_labels, allow_empty_assignments):
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_write_block, ds_in, ds_out,
                           blocking, block_id, node_labels, allow_empty_assignments)
                 for block_id in block_list]
        [t.result() for t in tasks]


def _load_assignments(path, key, n_threads):
    # if we have no key, this is a pickle file
    if key is None:
        assert os.path.split(path)[1].split('.')[-1] == 'pkl'
        with open(path, 'rb') as f:
            node_labels = pickle.load(f)
        assert isinstance(node_labels, dict)
    else:
        with vu.file_reader(path, 'r') as f:
            ds = f[key]
            assert ds.ndim in (1, 2)
            ds.n_threads = n_threads
            node_labels = ds[:]

            # this can happen if we only have a single label.
            # for some reason z5 returns this as int, not as array
            if isinstance(node_labels, int):
                node_labels = np.array([node_labels], dtype='uint64')

            # if we have 2d node_labels, these correspond to an assignment table
            # and we turn them into a dict for efficient downstream processing
            if node_labels.ndim == 2:
                if node_labels.shape[1] == 2:
                    node_labels = dict(zip(node_labels[:, 0], node_labels[:, 1]))
                elif node_labels.shape[0] == 2:
                    node_labels = dict(zip(node_labels[0, :], node_labels[1, :]))
                else:
                    raise ValueError("Invalid shape for 2d node labels")

    return node_labels


def _write_maxlabel(output_path, output_key, node_labels):
    if isinstance(node_labels, np.ndarray):
        max_id = int(node_labels.max())
    elif isinstance(node_labels, dict):
        max_id = int(np.max(list(node_labels.values())))
    else:
        raise AttributeError("Invalide type %s" % type(node_labels))
    with vu.file_reader(output_path) as f:
        f[output_key].attrs['maxId'] = max_id


def write(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("loading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read I/O config
    input_path = config['input_path']
    input_key = config['input_key']

    # check if we write in-place
    if 'output_path' in config:
        output_path = config['output_path']
        output_key = config['output_key']
        in_place = False
    else:
        in_place = True

    block_shape = config['block_shape']
    block_list = config['block_list']
    n_threads = config.get('threads_per_core', 1)
    allow_empty_assignments = config.get('allow_empty_assignments', False)

    # read node assignments
    assignment_path = config['assignment_path']
    assignment_key = config.get('assignment_key', None)
    fu.log("loading node labels from %s" % assignment_path)
    node_labels = _load_assignments(assignment_path, assignment_key, n_threads)

    offset_path = config.get('offset_path', None)

    # if we write in-place, we only need to open one file and one dataset
    if in_place:
        with vu.file_reader(input_path) as f:
            ds_in = f[input_key]
            ds_out = ds_in

            shape = ds_in.shape
            blocking = nt.blocking([0, 0, 0], list(shape), list(block_shape))

            if offset_path is None:
                _write(ds_in, ds_out, blocking, block_list, n_threads, node_labels, allow_empty_assignments)
            else:
                _write_with_offsets(ds_in, ds_out, blocking, block_list,
                                    n_threads, node_labels, offset_path, allow_empty_assignments)
        # write the max-label
        # for job 0
        if job_id == 0:
            _write_maxlabel(input_path, input_key, node_labels)

    else:
        # even if we do not write in-place, we might still write to the same output_file,
        # but different datasets
        # hdf5 does not like opening a file twice, so we need to check for this
        if input_path == output_path:
            with vu.file_reader(input_path) as f:
                ds_in = f[input_key]
                ds_out = f[output_key]

                shape = ds_in.shape
                blocking = nt.blocking([0, 0, 0], list(shape), list(block_shape))

                if offset_path is None:
                    _write(ds_in, ds_out, blocking, block_list, n_threads, node_labels, allow_empty_assignments)
                else:
                    _write_with_offsets(ds_in, ds_out, blocking, block_list,
                                        n_threads, node_labels, offset_path, allow_empty_assignments)
        else:
            with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
                ds_in = f_in[input_key]
                ds_out = f_out[output_key]

                shape = ds_in.shape
                blocking = nt.blocking([0, 0, 0], list(shape), list(block_shape))

                if offset_path is None:
                    _write(ds_in, ds_out, blocking, block_list, n_threads, node_labels, allow_empty_assignments)
                else:
                    _write_with_offsets(ds_in, ds_out, blocking, block_list,
                                        n_threads, node_labels, offset_path, allow_empty_assignments)
        # write the max-label
        # for job 0
        if job_id == 0:
            _write_maxlabel(output_path, output_key, node_labels)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    write(job_id, path)
