#! /usr/bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class BlockEdgeFeaturesBase(luigi.Task):
    """ Block edge feature base class
    """

    task_name = "block_edge_features"
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    output_path = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"offsets": None, "filters": None, "sigmas": None, "halo": [0, 0, 0],
                       "apply_in_2d": False, "channel_agglomeration": "mean"})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        subgraph_key = "s0/sub_graphs"
        output_key = "s0/sub_features"
        with vu.file_reader(self.graph_path, "r") as f:
            shape = tuple(f[subgraph_key].attrs["shape"])

        # require the output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(output_key, shape=shape, dtype="float64",
                              compression="gzip", chunks=tuple(block_shape))

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({"input_path": self.input_path, "input_key": self.input_key,
                       "labels_path": self.labels_path, "labels_key": self.labels_key,
                       "output_path": self.output_path, "block_shape": block_shape,
                       "graph_path": self.graph_path, "subgraph_key": subgraph_key,
                       "output_key": output_key})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class BlockEdgeFeaturesLocal(BlockEdgeFeaturesBase, LocalTask):
    """ BlockEdgeFeatures on local machine
    """
    pass


class BlockEdgeFeaturesSlurm(BlockEdgeFeaturesBase, SlurmTask):
    """ BlockEdgeFeatures on slurm cluster
    """
    pass


class BlockEdgeFeaturesLSF(BlockEdgeFeaturesBase, LSFTask):
    """ BlockEdgeFeatures on lsf cluster
    """
    pass


#
# Implementation
#


def _accumulate(input_path, input_key,
                labels_path, labels_key,
                graph_path, subgraph_key,
                output_path, output_key,
                block_list, offsets):

    fu.log("accumulate features without applying filters")
    with vu.file_reader(input_path, "r") as f:
        dtype = f[input_key].dtype
        input_dim = f[input_key].ndim

    if offsets is None:
        assert input_dim == 3, str(input_dim)
        fu.log("accumulate boundary map for type %s" % str(dtype))
        boundary_function = ndist.extractBlockFeaturesFromBoundaryMaps_uint8 if dtype == "uint8" else \
            ndist.extractBlockFeaturesFromBoundaryMaps_float32
        boundary_function(graph_path, subgraph_key,
                          input_path, input_key,
                          labels_path, labels_key,
                          block_list,
                          output_path, output_key,
                          increaseRoi=True)
    else:
        assert input_dim == 4, str(input_dim)
        fu.log("accumulate affinity map for type %s" % str(dtype))
        affinity_function = ndist.extractBlockFeaturesFromAffinityMaps_uint8 if dtype == "uint8" else \
            ndist.extractBlockFeaturesFromAffinityMaps_float32
        affinity_function(graph_path, subgraph_key,
                          input_path, input_key,
                          labels_path, labels_key,
                          block_list,
                          output_path, output_key,
                          offsets)
    # number of featres is 10 for both boundaries and affinities
    n_feats = 10
    return n_feats


def _accumulate_filter(input_, graph, labels, bb_local,
                       filter_name, sigma, ignore_label,
                       with_size, apply_in_2d):
    response = vu.apply_filter(input_, filter_name, sigma,
                               apply_in_2d=apply_in_2d)[bb_local]
    if response.ndim == 4:
        n_chan = response.shape[-1]
        assert response.shape[:-1] == labels.shape
        return np.concatenate([ndist.accumulateInput(graph, response[..., c], labels,
                                                     ignore_label,
                                                     with_size and c == n_chan-1,
                                                     response[..., c].min(),
                                                     response[..., c].max())
                               for c in range(n_chan)], axis=1)
    else:
        assert response.shape == labels.shape
        return ndist.accumulateInput(graph, response, labels,
                                     ignore_label, with_size,
                                     response.min(), response.max())


def _accumulate_block(block_id, blocking,
                      ds_in, ds_labels, ds_edges, ds_out,
                      filters, sigmas, halo, ignore_label,
                      apply_in_2d, channel_agglomeration):

    fu.log("start processing block %i" % block_id)
    chunk_pos = blocking.blockGridPosition(block_id)

    # load edges and construct the graph if this block has edges
    edges = ds_edges.read_chunk(chunk_pos)
    if edges is None:
        fu.log("block %i has no edges" % block_id)
        fu.log_block_success(block_id)
        return

    edges = edges.reshape((edges.size // 2, 2))
    graph = ndist.Graph(edges)

    shape = ds_labels.shape
    # get the bounding
    if sum(halo) > 0:
        block = blocking.getBlockWithHalo(block_id, halo)
        block_shape = block.outerBlock.shape
        bb_in = vu.block_to_bb(block.outerBlock)
        bb = vu.block_to_bb(block.innerBlock)
        bb_local = vu.block_to_bb(block.innerBlockLocal)
        # increase inner bounding box by 1 in posirive direction
        # in accordance with the graph extraction
        bb = tuple(slice(b.start,
                         min(b.stop + 1, sh)) for b, sh in zip(bb, shape))
        bb_local = tuple(slice(b.start,
                               min(b.stop + 1, bsh)) for b, bsh in zip(bb_local,
                                                                       block_shape))
    else:
        block = blocking.getBlock(block_id)
        bb = vu.block_to_bb(block)
        bb = tuple(slice(b.start,
                         min(b.stop + 1, sh)) for b, sh in zip(bb, shape))
        bb_in = bb
        bb_local = slice(None)

    input_dim = ds_in.ndim
    # TODO make choice of channels optional
    if input_dim == 4:
        bb_in = (slice(0, 3),) + bb_in

    input_ = vu.normalize(ds_in[bb_in])
    if input_dim == 4:
        assert channel_agglomeration is not None
        input_ = getattr(np, channel_agglomeration)(input_, axis=0)

    # load labels
    labels = ds_labels[bb]

    # TODO pre-smoothing ?!
    # accumulate the edge features
    edge_features = [_accumulate_filter(input_, graph, labels, bb_local,
                                        filter_name, sigma, ignore_label,
                                        filter_name == filters[-1] and sigma == sigmas[-1],
                                        apply_in_2d)
                     for filter_name in filters for sigma in sigmas]
    edge_features = np.concatenate(edge_features, axis=1)

    # save the features
    fu.log("saving feature result of shape %s" % str(edge_features.shape))
    ds_out.write_chunk(chunk_pos, edge_features.flatten(), True)
    fu.log_block_success(block_id)
    return edge_features.shape[1]


def _accumulate_with_filters(input_path, input_key,
                             labels_path, labels_key,
                             graph_path, subgraph_key,
                             output_path, output_key,
                             block_list, block_shape,
                             filters, sigmas, halo,
                             apply_in_2d, channel_agglomeration):

    fu.log("accumulate features with applying filters:")

    with vu.file_reader(input_path, "r") as f,\
            vu.file_reader(labels_path, "r") as fl,\
            vu.file_reader(graph_path, "r") as fg,\
            vu.file_reader(output_path) as fo:

        g = fg[subgraph_key]
        shape = g.attrs["shape"]
        ignore_label = g.attrs["ignore_label"]
        ds_edges = g["edges"]

        ds_in = f[input_key]
        ds_labels = fl[labels_key]
        ds_out = fo[output_key]

        blocking = nt.blocking([0, 0, 0], shape, block_shape)
        for block_id in block_list:
            n_feats = _accumulate_block(block_id, blocking,
                                        ds_in, ds_labels, ds_edges, ds_out,
                                        filters, sigmas, halo, ignore_label,
                                        apply_in_2d, channel_agglomeration)

    return n_feats


def block_edge_features(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, "r") as f:
        config = json.load(f)

    block_list = config["block_list"]
    input_path = config["input_path"]
    input_key = config["input_key"]
    labels_path = config["labels_path"]
    labels_key = config["labels_key"]
    output_path = config["output_path"]
    block_shape = config["block_shape"]
    graph_path = config["graph_path"]
    subgraph_key = config["subgraph_key"]
    output_key = config["output_key"]

    # offsets for accumulation of affinity maps
    offsets = config.get("offsets", None)
    filters = config.get("filters", None)
    sigmas = config.get("sigmas", None)
    apply_in_2d = config.get("apply_in_2d", False)
    halo = config.get("halo", [0, 0, 0])
    channel_agglomeration = config.get("channel_agglomeration", "mean")
    assert channel_agglomeration in ("mean", "max", "min", None)

    # check if we accumulate with filters
    with_filters = filters is not None

    with vu.file_reader(input_path, "r") as f:
        ndim = f[input_key].ndim

    fu.log(f"Offsets: {offsets}")
    has_channels = ndim == 4
    agglomerate_channels = False
    if has_channels and offsets is None:
        assert channel_agglomeration is not None
        agglomerate_channels = True

    if with_filters:
        assert offsets is None, "Filters and offsets are not supported"
        assert sigmas is not None, "Need sigma values"
        fu.log("Accumulate edge features with filters")
        n_feats = _accumulate_with_filters(input_path, input_key,
                                           labels_path, labels_key,
                                           graph_path, subgraph_key,
                                           output_path, output_key,
                                           block_list, block_shape,
                                           filters, sigmas, halo,
                                           apply_in_2d, channel_agglomeration)
    elif agglomerate_channels:
        fu.log("Accumulate edge features with channel agglomeration")
        filters = ["identity"]
        sigmas = [0]
        n_feats = _accumulate_with_filters(input_path, input_key,
                                           labels_path, labels_key,
                                           graph_path, subgraph_key,
                                           output_path, output_key,
                                           block_list, block_shape,
                                           filters, sigmas, halo,
                                           apply_in_2d, channel_agglomeration)
    else:
        fu.log("Accumulate edge features")
        n_feats = _accumulate(input_path, input_key,
                              labels_path, labels_key,
                              graph_path, subgraph_key,
                              output_path, output_key,
                              block_list, offsets)

    # we need to serialize the number of features for job 0
    if job_id == 0:
        with vu.file_reader(output_path, "a") as f:
            ds = f[output_key]
            ds.attrs["n_features"] = n_feats

    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    block_edge_features(job_id, path)
