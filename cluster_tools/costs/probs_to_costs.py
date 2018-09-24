#! /usr/bin/python

import os
import sys
import argparse
import pickle
import json

import numpy as np
import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


# NOTE we don't exclude the ignore label here, but ignore
# it in the graph extraction already
class ProbsToCostsBase(luigi.Task):
    """ ProbsToCosts base class
    """

    task_name = 'probs_to_costs'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    features_path = luigi.Parameter()
    features_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'invert_inputs': False, 'transform_to_costs': True,
                       'weight_edges': False, 'weighting_exponent': 1.,
                       'beta': 0.5})
        return config

    def run(self):
        # get the global config and init configs
        self.make_dirs()
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
            f.require_dataset(self.output_key, shape=(n_edges,), compression='gzip',
                              dtype='float32', chunks=(chunk_size,))

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'features_path': self.features_path, 'features_key': self.features_key})

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

def _transform_probabilities_to_costs(costs, beta=.5, edge_sizes=None,
                                      weighting_exponent=1.):
    """ Transform probabilities to costs (in-place)
    """
    p_min = 0.001
    p_max = 1. - p_min
    costs = (p_max - p_min) * costs + p_min
    # probabilities to costs, second term is boundary bias
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)
    # weight the costs with edge sizes, if they are given
    if edge_sizes is not None:
        assert len(edge_sizes) == len(costs)
        w = edge_sizes / edge_sizes.max()
        if weighting_exponent != 1.:
            w = w**weighting_exponent
        costs *= w
    return costs


def probs_to_costs(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    features_path = config['features_path']
    features_key = config['features_key']
    # config for cost transformations
    invert_inputs = config.get('invert_inputs', False)
    transform_to_costs = config.get('transform_to_costs', True)
    weight_edges = config.get('weight_edges', False)
    weighting_exponent = config.get('weighting_exponent', 1.)
    beta = config.get('beta', 0.5)

    n_threads = config['threads_per_job']

    fu.log("reading input from %s:%s" % (input_path, input_key))
    with vu.file_reader(input_path) as f:
        ds = f[input_key]
        ds.n_threads = n_threads
        # we might have 1d or 2d inputs, depending on input from features or random forest
        slice_ = slice(None) if ds.ndim == 1 else (slice(None), slice(0, 1))
        costs = ds[slice_].squeeze()

    # normalize to range 0, 1
    min_, max_ = costs.min(), costs.max()
    fu.log('input-range: %f %f' %  (min_, max_))
    fu.log('%f +- %f' % (costs.mean(), costs.std()))

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

        costs = _transform_probabilities_to_costs(costs, beta=beta,
                                                  edge_sizes=edge_sizes,
                                                  weighting_exponent=weighting_exponent)

    with vu.file_reader(output_path) as f:
        ds = f[output_key]
        ds.n_threads = n_threads
        ds[:] = costs

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    probs_to_costs(job_id, path)
