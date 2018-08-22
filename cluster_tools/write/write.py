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

    # input and output configs
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # the task we depend on
    dependency = luigi.TaskParameter()
    # we may have different write tasks,
    # so we need an identifier to keep them apart
    identifier = luigi.Parameter()
    offset_path = luigi.Parameter(default='')

    def requires(self):
        return self.dependency

    def _parse_log(self, log_path):
        log_path = self.input().path
        lines = fu.tail(log_path, 3)
        lines = [' '.join(ll.split()[2:]) for ll in lines]
        # check if this is a pickle file
        if lines[1].startswith("saving results to"):
            path = lines[1].split()[-1]
            assert os.path.exists(path)
            return path, None
        elif lines[0].startswith("saving results to"):
            path = lines[0].split()[-1]
            key = lines[1].split()[-1]
            return path, key
        else:
            raise RuntimeError("Could not parse log file.")

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        # require output dataset
        # TODO read chunks from config
        chunks = tuple(bs // 2 for bs in block_shape)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        n_threads = self.get_task_config().get('threads_per_core', 1)
        assignment_path, assignment_key = self._parse_log(self.input().path)
        # update the config with input and output paths and keys
        # as well as block shape
        config = {'input_path': self.input_path, 'input_key': self.input_key,
                  'output_path': self.output_path, 'output_key': self.output_key,
                  'block_shape': block_shape, 'n_threads': n_threads,
                  'assignment_path': assignment_path, 'assignment_key': assignment_key}
        if self.offset_path != '':
            config.update({'offset_path': self.offset_path})

        # get block list and jobs
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config, self.identifier)
        self.submit_jobs(n_jobs, self.identifier)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.identifier)
        self.check_jobs(n_jobs, self.identifier)


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


def _write_block_with_offsets(ds_in, ds_out, blocking, block_id,
                              node_labels, offsets):
    fu.log("start processing block %i" % block_id)
    off = offsets[block_id]
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    seg = ds_in[bb]
    seg[seg != 0] += off
    # choose the appropriate function for array or dictionary
    if isinstance(node_labels, np.ndarray):
        seg = nt.take(node_labels, seg)
    else:
        seg = nt.takeDict(node_labels, seg)
    ds_out[bb] = seg
    fu.log_block_success(block_id)


def _write_with_offsets(ds_in, ds_out, blocking, block_list,
                        n_threads, node_labels, offset_path):

    fu.log("loading offsets from %s" % offset_path)
    with open(offset_path) as f:
        offset_config = json.load(f)
        offsets = offset_config['offsets']
        empty_blocks = offset_config['empty_blocks']

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_write_block_with_offsets, ds_in, ds_out,
                           blocking, block_id, node_labels, offsets)
                 for block_id in block_list if block_id not in empty_blocks]
        [t.result() for t in tasks]


def _write_block(ds_in, ds_out, blocking, block_id, node_labels):
    # TODO should we lock the log ?
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    seg = ds_in[bb]
    # check if this block is empty and don't write if it is
    if np.sum(seg != 0) == 0:
        return

    # choose the appropriate function for array or dictionary
    if isinstance(node_labels, np.ndarray):
        seg = nt.take(node_labels, seg)
    else:
        # this copys the dict and hence is extremely RAM hungry
        this_labels = nt.unique(seg)
        this_assignment = {label: node_labels[label] for label in this_labels}
        seg = nt.takeDict(this_assignment, seg)

    ds_out[bb] = seg
    # TODO should we lock the log ?
    fu.log_block_success(block_id)


def _write(ds_in, ds_out, blocking, block_list,
           n_threads, node_labels):
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_write_block, ds_in, ds_out,
                           blocking, block_id, node_labels)
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
            ds.n_threads = n_threads
            node_labels = ds[:, 1] if ds.ndim == 2 else ds[:]
    return node_labels


def write(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("loading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read I/O config
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    block_shape = config['block_shape']
    block_list = config['block_list']
    n_threads = config['n_threads']

    # read node assignments
    assignment_path = config['assignment_path']
    assignment_key = config.get('assignment_key', None)
    fu.log("loading node labels from %s" % assignment_path)
    node_labels = _load_assignments(assignment_path, assignment_key, n_threads)

    offset_path = config.get('offset_path', None)

    # call write functions
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        shape = ds_in.shape
        blocking = nt.blocking([0, 0, 0], list(shape), list(block_shape))

        if offset_path is None:
            _write(ds_in, ds_out, blocking, block_list, n_threads, node_labels)
        else:
            _write_with_offsets(ds_in, ds_out, blocking, block_list,
                                n_threads, node_labels, offset_path)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    write(job_id, path)
