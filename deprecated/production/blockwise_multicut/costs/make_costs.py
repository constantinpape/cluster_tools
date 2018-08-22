#! /usr/bin/python

import os
import time
import argparse
import json
import subprocess
import cremi_tools.segmentation as cseg

import z5py
import luigi


class CostsTask(luigi.Task):
    """
    Transform features to multicut costs
    """

    features_path = luigi.Parameter()
    graph_path = luigi.Parameter()
    out_path = luigi.Parameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def run(self):
        from production import util

        # copy the script to the temp folder and replace the shebang
        script_path = os.path.join(self.tmp_folder, 'make_costs.py')
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'make_costs.py'),
                              script_path)

        with open(self.config_path) as f:
            config = json.load(f)
            beta = config.get('beta', 0.5)
            weighting_exponent = config.get('weighting_exponent', 1.)
            weight_edges = config.get('weight_multicut_edges', False)

        # write job config
        job_config = {'beta': beta, 'weight_edges': weight_edges,
                      'weighting_exponent': weighting_exponent}
        config_path = os.path.join(self.tmp_folder, 'make_costs_config.json')
        with open(config_path, 'w') as f:
            json.dump(job_config, f)

        # submit job
        command = '%s %s %s %s %s %s' % (script_path, self.features_path,
                                         self.graph_path, self.out_path,
                                         config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_costs')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_costs')
        bsub_command = 'bsub -J costs -We %i -o %s -e %s \'%s\'' % (self.time_estimate,
                                                                    log_file, err_file,
                                                                    command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)
            util.wait_for_jobs('papec')

        try:
            with open(self.output().path) as f:
                json.load(f)['t']
            success = True
        except Exception:
            success = False

        if not success:
            raise RuntimeError("CostsTask failed")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'costs.log'))


# TODO implement with random forest
# TODO multi-threaded
def make_costs(features_path,
               graph_path,
               output_path,
               config_path,
               tmp_folder,
               invert_inputs=True):

    t0 = time.time()
    features_key = 'features'
    graph_key = 'graph'
    out_key = 'costs'

    with open(config_path) as f:
        config = json.load(f)
        beta = config.get('beta', 0.5)
        weighting_exponent = config.get('weighting_exponent', 1.)
        weight_edges = config.get('weight_multicut_edges', False)

    # find the ignore edges
    ds_graph = z5py.File(graph_path)[graph_key]
    n_edges = ds_graph['edges'].shape[0]
    uv_ids = ds_graph['edges'][:]
    ignore_edges = (uv_ids == 0).any(axis=1)

    # get the multicut edge costs from mean affinities
    feat_ds = z5py.File(features_path)[features_key]
    # we need to invert the features accumulated from affinities
    if invert_inputs:
        costs = 1. - feat_ds[:, 0].squeeze()

    if weight_edges:
        # TODO the edge sizes might not be hardcoded to this feature
        # id in the future
        edge_sizes = feat_ds[:, -1].squeeze()
        # set edge sizes of ignore edges to 1 (we don't want them to influence the weighting)
        edge_sizes[ignore_edges] = 1
    else:
        edge_sizes = None

    costs = cseg.transform_probabilities_to_costs(costs, beta=beta,
                                                  edge_sizes=edge_sizes,
                                                  weighting_exponent=weighting_exponent)
    # set weights of ignore edges to be maximally repulsive
    costs[ignore_edges] = -100
    f_out = z5py.File(output_path)
    chunks = (min(262144, n_edges),)
    ds_out = f_out.require_dataset(out_key, shape=(n_edges,), dtype='float32',
                                   chunks=chunks, compression='raw')
    ds_out[:] = costs.astype('float32')

    res_file = os.path.join(tmp_folder, 'costs.log')
    with open(res_file, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('features_path', type=str)
    parser.add_argument('graph_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('config_path', type=str)
    parser.add_argument('tmp_folder', type=str)

    args = parser.parse_args()
    make_costs(args.features_path,
               args.graph_path,
               args.out_path,
               args.config_path,
               args.tmp_folder)
