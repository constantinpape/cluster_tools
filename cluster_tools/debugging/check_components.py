#! /usr/bin/python

import os
import sys
import json
from concurrent import futures
from collections import ChainMap

import luigi
import numpy as np
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class CheckComponentsBase(luigi.Task):
    """ CheckComponents base class
    """

    allow_retry = False
    task_name = 'check_components'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    chunks = luigi.ListParameter()
    number_of_labels = luigi.IntParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, _, _ = self.global_config_values()
        self.init(shebang)

        # we don't need any additional config besides the paths
        config = self.get_task_config()
        config.update({"input_path": self.input_path, "input_key": self.input_key,
                       "output_path": self.output_path, "output_key": self.output_key,
                       "block_shape": block_shape, "chunks": self.chunks,
                       "n_labels": self.number_of_labels})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class CheckComponentsLocal(CheckComponentsBase, LocalTask):
    """
    CheckComponents on local machine
    """
    pass


class CheckComponentsSlurm(CheckComponentsBase, SlurmTask):
    """
    CheckComponents on slurm cluster
    """
    pass


class CheckComponentsLSF(CheckComponentsBase, LSFTask):
    """
    CheckComponents on lsf cluster
    """
    pass


#
# Implementation
#


def _check_components_impl(ds, max_chunks_per_label, n_threads,
                           number_of_labels):

    chunk_size = ds.chunks[0]
    n_chunks = number_of_labels // chunk_size + 1

    def check_labels_in_chunk(chunk_id):
        # TODO this does not lift gil atm
        mapping = ndist.readBlockMapping(ds.path, (chunk_id,))
        if not mapping:
            return {}

        violating_ids = {label_id: len(blocks)
                         for label_id, blocks in mapping.items()
                         if len(blocks) > max_chunks_per_label}
        return violating_ids

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(check_labels_in_chunk, chunk_id)
                 for chunk_id in range(n_chunks)]
        results = [t.result() for t in tasks]
        results = [res for res in results if res]

    results = dict(ChainMap(*results))
    ids = np.array(list(results.keys()))
    blocks_per_id = np.array(list(results.values()))

    violating_ids = np.concatenate([ids[:, None],
                                    blocks_per_id[:, None]], axis=1)

    return violating_ids


def check_components(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    block_shape = config['block_shape']
    chunks = config['chunks']
    n_labels = config['n_labels']

    chunks_per_block = [bs // ch for bs, ch in zip(block_shape, chunks)]
    max_chunks_per_label = np.prod(chunks_per_block)
    # TODO don't hard-code assertion to special case for [512, 512, 50], [256, 256, 25]
    assert max_chunks_per_label == 8

    n_threads = config.get('threads_per_job', 1)

    ds_in = vu.file_reader(input_path)[input_key]
    violating_ids = _check_components_impl(ds_in, max_chunks_per_label,
                                           n_threads, n_labels)

    if violating_ids.size > 0:
        fu.log("have %i violationg_ids" % violating_ids.shape[0])
        vchunks = (min(10000, violating_ids.shape[0]), 2)
        with vu.file_reader(output_path) as f:
            f.create_dataset(output_key, data=violating_ids, chunks=vchunks)
    else:
        fu.log("no violating ids")

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    check_components(job_id, path)
