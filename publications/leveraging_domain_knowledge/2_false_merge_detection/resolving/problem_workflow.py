#! /bin/python

import os
import sys
import json

import luigi
import z5py
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

# FIXME
# need to put this on the python path
# folder = os.path.split(os.path.abspath(__file__))[0]
# top = os.path.abspath(os.path.join(folder, '..'))
top = '/g/kreshuk/pape/Work/my_projects/lifted_priors/fib25_experiments/segmentation'
sys.path.append(top)
from resolving import edges_to_problem, combine_edges_and_costs


#
# Node Label Tasks
#


class LiftedEdgesBase(luigi.Task):
    """ LiftedEdges base class
    """

    task_name = 'lifted_edges'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    path = luigi.Parameter()
    exp_path = luigi.Parameter()
    precision = luigi.FloatParameter()
    recall = luigi.FloatParameter()

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'path': self.path,
                       'exp_path': self.exp_path,
                       'precision': self.precision,
                       'recall': self.recall})

        # prime and run the jobs
        n_jobs = 1
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class LiftedEdgesLocal(LiftedEdgesBase, LocalTask):
    """ LiftedEdges on local machine
    """
    pass


class LiftedEdgesSlurm(LiftedEdgesBase, SlurmTask):
    """ LiftedEdges on slurm cluster
    """
    pass


class LiftedEdgesLSF(LiftedEdgesBase, LSFTask):
    """ LiftedEdges on lsf cluster
    """
    pass


#
# Implementation
#


def full_lifted_problem(path, n_threads):
    # we make softlinks to graph and local costs, etc.
    ref_path = '/g/kreshuk/data/FIB25/exp_data/mc.n5'
    attrs = z5py.File(ref_path).attrs

    f = z5py.File(path)
    for k, v in attrs.items():
        f.attrs[k] = v

    g = f.require_group('s0')
    to_link = ['costs', 'graph', 'sub_graphs']
    for t in to_link:
        src = os.path.join(ref_path, 's0', t)
        dst = os.path.join(g.path, t)
        if os.path.exists(dst):
            continue
        os.symlink(src, dst)

    in_group = 'resolving/oracle'
    out_key_uvs = 's0/lifted_nh_resolving'
    out_key_costs = 's0/lifted_costs_resolving'
    combine_edges_and_costs(path, in_group, path,
                            out_key_uvs, out_key_costs, n_threads)


def get_max_costs(factor=1):
    path = '/g/kreshuk/data/FIB25/exp_data/mc.n5'
    f = z5py.File(path)
    ds = f['s0/costs']
    ds.n_threads = 8
    costs = ds[:]

    max_abs_cost = max([abs(costs.min()),
                        abs(costs.max())])

    max_attractive = factor * max_abs_cost
    max_repulsive = -1 * factor * max_abs_cost
    return max_attractive, max_repulsive


def lifted_edges(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    key_in = 'resolving/edges_and_indicators'
    key_uv = 'resolving/oracle/uvs'
    key_costs = 'resolving/oracle/costs'

    exp_path = config['exp_path']
    path = config['path']
    precision = config['precision']
    recall = config['recall']
    n_threads = config.get('threads_per_job', 1)

    max_attractive, max_repulsive = get_max_costs()
    fu.log("Making costs with max attractive / repulsive values: %f / %f" % (max_attractive,
                                                                             max_repulsive))

    # extract subproblems
    edges_to_problem(path, exp_path,
                     key_in, key_uv, key_costs,
                     precision, recall,
                     max_attractive, max_repulsive,
                     n_threads)

    # make full lifted problem
    full_lifted_problem(exp_path, n_threads)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    lifted_edges(job_id, path)
