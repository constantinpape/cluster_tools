#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import nifty
import nifty.graph.rag as nrag
from vigra.analysis import relabelConsecutive

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.segmentation_utils as su
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Agglomerate Tasks
#

# TODO it would be nice to be able to change the block shape compared to ws task
# so that we can agglomerate block boundaries.
# However, I am not sure how this interacts with the id-offsets, so haven't
# implemented this yet.
class AgglomerateBase(luigi.Task):
    """ Agglomerate base class
    """

    task_name = 'agglomerate'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    have_ignore_label = luigi.BoolParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # parameter:
        # use_mala_agglomeration: whether to use thresholding based mala agglomeration
        #                         or element number based agglomerative clustering
        # threshold: threshold up to which to agglomerate (mala) or fraction of nodes
        #            after agglomeration (agglomerative clustering)
        # size_regularizer: size regularizer in agglomerative clustering (wardness)
        # invert_inputs: do we need to invert the inputs?
        # offsets: offsets for affinities, set to None for boundaries
        config = LocalTask.default_task_config()
        config.update({'use_mala_agglomeration': True, 'threshold': .9,
                       'size_regularizer': .5, 'invert_inputs': False,
                       'offsets': None})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)
        if len(shape) == 4:
            shape = shape[1:]

        # load the agglomerate config
        config = self.get_task_config()

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape, 'have_ignore_label': self.have_ignore_label})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        self._write_log('scheduling %i blocks to be processed' % len(block_list))
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class AgglomerateLocal(AgglomerateBase, LocalTask):
    """
    Agglomerate on local machine
    """
    pass


class AgglomerateSlurm(AgglomerateBase, SlurmTask):
    """
    Agglomerate on slurm cluster
    """
    pass


class AgglomerateLSF(AgglomerateBase, LSFTask):
    """
    Agglomerate on lsf cluster
    """
    pass


#
# Implementation
#

def _agglomerate_block(blocking, block_id, ds_in, ds_out, config):
    fu.log("start processing block %i" % block_id)
    have_ignore_label = config['have_ignore_label']
    use_mala_agglomeration = config.get('use_mala_agglomeration', True)
    threshold = config.get('threshold', 0.9)
    size_regularizer = config.get('size_regularizer', .5)
    invert_inputs = config.get('invert_inputs', False)
    offsets = config.get('offsets', None)

    bb = vu.block_to_bb(blocking.getBlock(block_id))
    # load the segmentation / output
    seg = ds_out[bb]

    # check if this block is empty
    if np.sum(seg) == 0:
        fu.log_block_success(block_id)
        return

    # load the input data
    ndim_in = ds_in.ndim
    if ndim_in == 4:
        assert offsets is not None
        assert len(offsets) <= ds_in.shape[0]
        bb_in = (slice(0, len(offsets)),) + bb
        input_ = vu.normalize(ds_in[bb_in])
    else:
        assert offsets is None
        input_ = vu.normalize(ds_in[bb])

    if invert_inputs:
        input_ = 1. - input_

    id_offset = int(seg[seg != 0].min())

    # relabel the segmentation
    _, max_id, _ = relabelConsecutive(seg, out=seg, keep_zeros=True, start_label=1)
    seg = seg.astype('uint32')

    # construct rag
    rag = nrag.gridRag(seg, numberOfLabels=max_id + 1,
                       numberOfThreads=1)

    # extract edge features
    if offsets is None:
        edge_features = nrag.accumulateEdgeMeanAndLength(rag, input_, numberOfThreads=1)
    else:
        edge_features = nrag.accumulateAffinityStandartFeatures(rag, input_, offsets,
                                                                numberOfThreads=1)
    edge_features, edge_sizes = edge_features[:, 0], edge_features[:, -1]
    uv_ids = rag.uvIds()
    # set edges to ignore label to be maximally repulsive
    if have_ignore_label:
        ignore_mask = (uv_ids == 0).any(axis=1)
        edge_features[ignore_mask] = 1

    # build undirected graph
    n_nodes = rag.numberOfNodes
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)

    if use_mala_agglomeration:
        node_labels = su.mala_clustering(graph, edge_features,
                                         edge_sizes, threshold)
    else:
        node_ids, node_sizes = np.unique(seg, return_counts=True)
        if node_ids[0] != 0:
            node_sizes = np.concatenate([np.array([0]), node_sizes])
        n_stop = int(threshold * n_nodes)
        node_labels = su.agglomerative_clustering(graph, edge_features,
                                                  node_sizes, edge_sizes,
                                                  n_stop, size_regularizer)

    # run clusteting
    node_labels, max_id, _ = relabelConsecutive(node_labels, start_label=1, keep_zeros=True)

    fu.log("reduced number of labels from %i to %i" % (n_nodes, max_id + 1))

    # project node labels back to segmentation
    seg = nrag.projectScalarNodeDataToPixels(rag, node_labels, numberOfThreads=1)
    seg = seg.astype('uint64')
    # add offset back to segmentation
    seg[seg != 0] += id_offset

    ds_out[bb] = seg
    # log block success
    fu.log_block_success(block_id)


def agglomerate(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']
    shape = list(vu.get_shape(input_path, input_key))
    if len(shape) == 4:
        shape = shape[1:]

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    # read the output config
    output_path = config['output_path']
    output_key = config['output_key']

    # get the blocking
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        assert ds_in.ndim in (3, 4)
        ds_out = f_out[output_key]
        assert ds_out.ndim == 3
        for block_id in block_list:
            _agglomerate_block(blocking, block_id, ds_in, ds_out, config)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    agglomerate(job_id, path)
