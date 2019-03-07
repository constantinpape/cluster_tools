#! /usr/bin/python

import os
import sys
import json
import pickle

import luigi
import numpy as np
import nifty.tools as nt
from affogato.affinities import compute_embedding_distances

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Block-wise embedding distance tasks
#

class EmbeddingDistancesBase(luigi.Task):
    """ EmbeddingDistances base class
    """

    task_name = 'embedding_distances'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    path_dict = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    offsets = luigi.ListParameter(default=[[-1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]])
    norm = luigi.Parameter(default='l2')
    threshold = luigi.FloatParameter(default=None)
    threshold_mode = luigi.Parameter(default='greater')
    dependency = luigi.TaskParameter(default=DummyTask())

    threshold_modes = ('greater', 'less', 'equal')
    norms = ('l2', 'cosine')

    def requires(self):
        return self.dependency

    def _validate_paths(self):
        shape = None

        with open(self.path_dict) as f:
            path_dict = json.load(f)

        for path in sorted(path_dict):
            key = path_dict[path]
            assert os.path.exists(path)
            with vu.file_reader(path, 'r') as f:
                assert key in f
                ds = f[key]
                if shape is None:
                    shape = ds.shape
                else:
                    # TODO support multi-channel inputs and then only check that
                    # spatial shapes agree
                    assert ds.shape == shape
        return shape

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        assert self.norm in self.norms
        if self.threshold is not None:
            assert self.threshold_mode in self.threshold_modes

        shape = self._validate_paths()
        config = self.get_task_config()
        config.update({'path_dict': self.path_dict,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'offsets': self.offsets,
                       'block_shape': block_shape,
                       'norm': self.norm,
                       'threshold': self.threshold,
                       'threshold_mode': self.threshold_mode})

        n_channels = len(self.offsets)
        chunks = tuple(min(bs // 2, sh) for bs, sh in zip(block_shape, shape))

        out_shape = (n_channels,) + shape
        out_chunks = (1,) + chunks

        # make output dataset
        compression = config.pop('compression', 'gzip')
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=out_shape, dtype='float32',
                              compression=compression, chunks=out_chunks)

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class EmbeddingDistancesLocal(EmbeddingDistancesBase, LocalTask):
    """
    EmbeddingDistances on local machine
    """
    pass


class EmbeddingDistancesSlurm(EmbeddingDistancesBase, SlurmTask):
    """
    EmbeddingDistances on slurm cluster
    """
    pass


class EmbeddingDistancesLSF(EmbeddingDistancesBase, LSFTask):
    """
    EmbeddingDistances on lsf cluster
    """
    pass


def _embedding_distances_block(block_id, blocking,
                               input_datasets, ds, offsets,
                               norm):
    fu.log("start processing block %i" % block_id)
    halo = np.max(np.abs(offsets), axis=0)

    block = blocking.getBlockWithHalo(block_id, halo.tolist())
    outer_bb = vu.block_to_bb(block.outerBlock)
    inner_bb = (slice(None),) + vu.block_to_bb(block.innerBlock)
    local_bb = (slice(None),) + vu.block_to_bb(block.innerBlockLocal)

    bshape = tuple(ob.stop - ob.start for ob in outer_bb)
    # TODO support multi-channel input data
    n_inchannels = len(input_datasets)
    in_shape = (n_inchannels,) + bshape
    in_data = np.zeros(in_shape, dtype='float32')

    for chan, inds in enumerate(input_datasets):
        in_data[chan] = inds[outer_bb]

    # TODO support thresholding the embedding before distance caclulation
    distance = compute_embedding_distances(in_data, offsets, norm)
    ds[inner_bb] = distance[local_bb]

    fu.log_block_success(block_id)


def embedding_distances(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    path_dict = config['path_dict']
    output_path = config['output_path']
    output_key = config['output_key']
    block_list = config['block_list']
    block_shape = config['block_shape']
    offsets = config['offsets']
    norm = config['norm']

    # TODO support thresholding
    threshold = config['threshold']
    threshold_mode = config['threshold_mode']
    assert threshold is None

    with open(path_dict) as f:
        path_dict = json.load(f)

    input_datasets = []
    for path in sorted(path_dict):
        input_datasets.append(vu.file_reader(path, 'r')[path_dict[path]])

    with vu.file_reader(output_path) as f:

        ds = f[output_key]

        shape = ds.shape[1:]
        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)
        [_embedding_distances_block(block_id, blocking, input_datasets, ds, offsets, norm)
         for block_id in block_list]

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    embedding_distances(job_id, path)
