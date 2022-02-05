#! /bin/python

import os
import sys
import json

import luigi
import vigra
import nifty
from elf.segmentation.lifted_multicut import get_lifted_multicut_solver

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

#
# Lifted Multicut Tasks
#


class SolveLiftedGlobalBase(luigi.Task):
    """ SolveLiftedGlobal base class
    """

    task_name = "solve_lifted_global"
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    problem_path = luigi.Parameter()
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    scale = luigi.IntParameter()
    lifted_prefix = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"agglomerator": "kernighan-lin",
                       "time_limit_solver": None})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({"assignment_path": self.assignment_path, "assignment_key": self.assignment_key,
                       "scale": self.scale, "problem_path": self.problem_path,
                       "lifted_prefix": self.lifted_prefix})

        # prime and run the job
        prefix = "s%i" % self.scale
        self.prepare_jobs(1, None, config, prefix)
        self.submit_jobs(1, prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1, prefix)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + "_s%i.log" % self.scale))


class SolveLiftedGlobalLocal(SolveLiftedGlobalBase, LocalTask):
    """ SolveLiftedGlobal on local machine
    """
    pass


class SolveLiftedGlobalSlurm(SolveLiftedGlobalBase, SlurmTask):
    """ SolveLiftedGlobal on slurm cluster
    """
    pass


class SolveLiftedGlobalLSF(SolveLiftedGlobalBase, LSFTask):
    """ SolveLiftedGlobal on lsf cluster
    """
    pass


#
# Implementation
#


def solve_lifted_global(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    # path to the reduced problem
    problem_path = config["problem_path"]
    # path where the node labeling shall be written
    assignment_path = config["assignment_path"]
    assignment_key = config["assignment_key"]

    lifted_prefix = config["lifted_prefix"]
    scale = config["scale"]
    agglomerator_key = config["agglomerator"]
    n_threads = config["threads_per_job"]
    time_limit = config.get("time_limit_solver", None)

    fu.log("using agglomerator %s" % agglomerator_key)
    solver = get_lifted_multicut_solver(agglomerator_key)

    with vu.file_reader(problem_path) as f:
        group = f["s%i" % scale]
        graph_group = group["graph"] if scale == 0 else group["graph_lmc"]
        ignore_label = graph_group.attrs["ignore_label"]

        ds = graph_group["edges"]
        ds.n_threads = n_threads
        uv_ids = ds[:]
        n_edges = len(uv_ids)
        n_nodes = int(uv_ids.max()) + 1

        if scale > 0:
            ds = group["node_labeling_lmc"]
            ds.n_threads = n_threads
            initial_node_labeling = ds[:]

        ds = group["costs"] if scale == 0 else group["costs_lmc"]
        ds.n_threads = n_threads
        costs = ds[:]
        assert len(costs) == n_edges, f"{len(costs)}, {n_edges}"

        ds = group["lifted_nh_%s" % lifted_prefix]
        ds.n_threads = n_threads
        lifted_uvs = ds[:]

        ds = group["lifted_costs_%s" % lifted_prefix]
        ds.n_threads = n_threads
        lifted_costs = ds[:]

    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    fu.log("start agglomeration")
    node_labeling = solver(graph, costs,
                           lifted_uvs, lifted_costs,
                           n_threads=n_threads,
                           time_limit=time_limit)
    fu.log("finished agglomeration")

    if scale > 0:
        # get the labeling of initial nodes
        initial_node_labeling = node_labeling[initial_node_labeling]
    else:
        initial_node_labeling = node_labeling
    n_nodes = len(initial_node_labeling)

    # make sure zero is mapped to 0 if we have an ignore label
    if ignore_label and initial_node_labeling[0] != 0:
        new_max_label = int(node_labeling.max() + 1)
        initial_node_labeling[initial_node_labeling == 0] = new_max_label
        initial_node_labeling[0] = 0

    # make node labeling consecutive
    vigra.analysis.relabelConsecutive(initial_node_labeling, start_label=1, keep_zeros=True,
                                      out=initial_node_labeling)

    # write assignments
    node_shape = (n_nodes,)
    chunks = (min(n_nodes, 524288),)
    with vu.file_reader(assignment_path) as f:
        ds = f.require_dataset(assignment_key, dtype="uint64",
                               shape=node_shape,
                               chunks=chunks,
                               compression="gzip")
        ds.n_threads = n_threads
        ds[:] = initial_node_labeling

    fu.log("saving results to %s:%s" % (assignment_path, assignment_key))
    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    solve_lifted_global(job_id, path)
