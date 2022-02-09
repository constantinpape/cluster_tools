#! /usr/bin/python

import os
import sys
import json

import numpy as np
import luigi
from elf.segmentation.multicut import transform_probabilities_to_costs

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


# NOTE we don't exclude the ignore label here, but ignore it in the graph extraction already
class ProbsToCostsBase(luigi.Task):
    """ ProbsToCosts base class
    """

    task_name = "probs_to_costs"
    src_file = os.path.abspath(__file__)
    allow_retry = False
    # modes which can be used to mask edges
    # that connect nodes of a certain type (specified via `node_label_dict`)
    # - ignore: set all edges connecting to a node with label to be maximally repulsive
    # - isolate : set all edges between nodes with label to be maximally attractive,
    #             all edges between nodes with and w/o label to be maximally attractive
    # - ignore_transition: set transitions between labels to be max repulsive
    label_modes = ("ignore", "isolate", "ignore_transition")

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    features_path = luigi.Parameter()
    features_key = luigi.Parameter()
    dependency = luigi.TaskParameter()
    node_label_dict = luigi.DictParameter(default={})

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"invert_inputs": False, "transform_to_costs": True,
                       "weight_edges": False, "weighting_exponent": 1.,
                       "beta": 0.5})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        with vu.file_reader(self.input_path) as f:
            n_edges = f[self.input_key].shape[0]
        # chunk size = 64**3
        chunk_size = min(262144, n_edges)

        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=(n_edges,), compression="gzip",
                              dtype="float32", chunks=(chunk_size,))

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({"input_path": self.input_path, "input_key": self.input_key,
                       "output_path": self.output_path, "output_key": self.output_key,
                       "features_path": self.features_path, "features_key": self.features_key})

        # check if we have additional node labels and update the config accordingly
        if self.node_label_dict:
            assert all(mode in self.label_modes
                       for mode in self.node_label_dict), str(list(self.node_label_dict.keys()))
            config.update({"node_labels": dict(self.node_label_dict)})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class ProbsToCostsLocal(ProbsToCostsBase, LocalTask):
    """ ProbsToCosts on local machine
    """
    pass


class ProbsToCostsSlurm(ProbsToCostsBase, SlurmTask):
    """ ProbsToCosts on slurm cluster
    """
    pass


class ProbsToCostsLSF(ProbsToCostsBase, LSFTask):
    """ ProbsToCosts on lsf cluster
    """
    pass


#
# Implementation
#

def _apply_node_labels(costs, uv_ids, mode, labels,
                       max_repulsive, max_attractive):
    # TODO for now we assume binary node labeling,
    # but of course we could also have something more fancy with
    # multiple label ids
    n_nodes = len(labels)
    max_node_id = int(uv_ids.max())
    assert max_node_id + 1 <= n_nodes, "%i, %i" % (max_node_id, n_nodes)
    with_label = np.arange(n_nodes, dtype="uint64")[labels > 0]
    fu.log("number of nodes with label %i / %i" % (len(with_label), n_nodes))
    if mode == "ignore":
        fu.log("Node-label mode: ignore")
        # ignore mode: set all edges that connect to a node with label to max repulsive
        edges_with_label = np.isn(uv_ids, with_label)
        edges_with_label = edges_with_label.any(axis=1)
        costs[edges_with_label] = max_repulsive
    elif mode == "isolate":
        # isolate mode: set all edges that connect to a node with label to node without label to max repulsive
        fu.log("Node-label mode: isolate")
        # ignore mode: set all edges that connect two node with label to max attractive
        edges_with_label = np.in1d(uv_ids, with_label).reshape(uv_ids.shape)
        label_sum = edges_with_label.sum(axis=1)
        att_edges = label_sum == 2
        rep_edges = label_sum == 1
        fu.log("number of attractive edges: %i / %i" % (att_edges.sum(), len(att_edges)))
        fu.log("number of repulsive edges: %i / %i" % (rep_edges.sum(), len(rep_edges)))
        costs[att_edges] = max_attractive
        costs[rep_edges] = max_repulsive
    elif mode == "ignore_transition":
        fu.log("Node-label mode: ignore_transition")
        labels_mapped_to_edges = labels[uv_ids]
        transition = labels_mapped_to_edges[:, 0] != labels_mapped_to_edges[:, 1]
        costs[transition] = max_repulsive
        fu.log("number of repulsive edges: %i / %i" % (transition.sum(), len(transition)))
    else:
        raise RuntimeError("Invalid label mode: %s" % mode)
    return costs


def probs_to_costs(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    input_path = config["input_path"]
    input_key = config["input_key"]
    output_path = config["output_path"]
    output_key = config["output_key"]
    features_path = config["features_path"]
    features_key = config["features_key"]
    # config for cost transformations
    invert_inputs = config.get("invert_inputs", False)
    transform_to_costs = config.get("transform_to_costs", True)
    weight_edges = config.get("weight_edges", False)
    weighting_exponent = config.get("weighting_exponent", 1.)
    beta = config.get("beta", 0.5)

    # additional node labels
    node_labels = config.get("node_labels", None)

    n_threads = config["threads_per_job"]

    fu.log("reading input from %s:%s" % (input_path, input_key))
    with vu.file_reader(input_path) as f:
        ds = f[input_key]
        ds.n_threads = n_threads
        # we might have 1d or 2d inputs, depending on input from features or random forest
        slice_ = slice(None) if ds.ndim == 1 else (slice(None), slice(0, 1))
        costs = ds[slice_].squeeze()

    # normalize to range 0, 1
    min_, max_ = costs.min(), costs.max()
    fu.log("input-range: %f %f" % (min_, max_))
    fu.log("%f +- %f" % (costs.mean(), costs.std()))

    if invert_inputs:
        fu.log("inverting probability inputs")
        costs = 1. - costs

    if transform_to_costs:
        fu.log("converting probability inputs to costs")
        if weight_edges:
            fu.log("weighting edges by size")
            # the edge sizes are at the last feature index
            with vu.file_reader(features_path) as f:
                ds = f[features_key]
                n_features = ds.shape[1]
                ds.n_threads = n_threads
                edge_sizes = ds[:, n_features-1:n_features].squeeze()
        else:
            fu.log("no edge weighting")
            edge_sizes = None

        costs = transform_probabilities_to_costs(costs, beta=beta,
                                                 edge_sizes=edge_sizes,
                                                 weighting_exponent=weighting_exponent)

        # adjust edges of nodes with labels if given
        if node_labels is not None:
            fu.log("have node labels")
            max_repulsive = 5 * costs.min()
            max_attractive = 5 * costs.max()
            fu.log("maximally attractive edge weight %f" % max_attractive)
            fu.log("maximally repulsive edge weight %f" % max_repulsive)
            with vu.file_reader(features_path, "r") as f:
                ds = f["s0/graph/edges"]
                ds.n_threads = n_threads
                uv_ids = ds[:]
            for mode, path_key in node_labels.items():
                path, key = path_key
                fu.log("applying node labels with mode %s from %s:%s" % (mode, path, key))
                with vu.file_reader(path, "r") as f:
                    ds = f[key]
                    ds.n_threads = n_threads
                    labels = ds[:]
                costs = _apply_node_labels(costs, uv_ids, mode, labels,
                                           max_repulsive, max_attractive)

    with vu.file_reader(output_path) as f:
        ds = f[output_key]
        ds.n_threads = n_threads
        ds[:] = costs

    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    probs_to_costs(job_id, path)
