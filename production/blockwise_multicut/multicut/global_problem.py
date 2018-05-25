#! /usr/bin/python

import time
import os
import argparse
import subprocess
import json
from shutil import rmtree

import z5py
import nifty
import luigi
import cremi_tools.segmentation as cseg

# TODO support more agglomerators
AGGLOMERATORS = {"multicut_kl": cseg.Multicut("kernighan-lin")}


class GlobalProblemTask(luigi.Task):
    """
    Solve the global reduced problem
    """

    path = luigi.Parameter()
    out_key = luigi.Parameter()
    max_scale = luigi.IntParameter()
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
        script_path = os.path.join(self.tmp_folder, 'global_problem.py')
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'global_problem.py'),
                              script_path)

        with open(self.config_path) as f:
            config = json.load(f)
            n_threads = config['n_threads']
            # TODO support computation with roi
            if 'roi' in config:
                roi = config['roi']
            else:
                roi = None

        # prepare the job config
        job_config = {'n_threads': n_threads}
        config_path = os.path.join(self.tmp_folder,
                                   'global_problem_config.json')
        with open(config_path, 'w') as f:
            json.dump(job_config, f)

        command = '%s %s %s %i %s %s' % (script_path, self.path,
                                         self.out_key, self.max_scale,
                                         config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder,
                                'logs', 'log_global_problem')
        err_file = os.path.join(self.tmp_folder,
                                'error_logs', 'err_global_problem')
        bsub_command = ('bsub -n %i -J global_problem ' % n_threads +
                        '-We %i -o %s -e %s \'%s\'' % (self.time_estimate,
                                                       log_file, err_file, command))
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)
            util.wait_for_jobs('papec')

        # check the output
        try:
            res_path = os.path.join(self.tmp_folder, 'global_problem.log')
            with open(res_path) as f:
                res = json.load(f)
                t = res['t']
            print("Global problem finished in", t, "s")
            success = True
        except Exception:
            success = False
            # clean up the output
            rmtree(os.path.join(self.path, self.out_key))

        # write output file if we succeed, otherwise write partial
        # success to different file and raise exception
        if not success:
            raise RuntimeError("GlobalProblemTask failed")

    def output(self):
        res_path = os.path.join(self.path, self.out_key)
        return luigi.LocalTarget(res_path)


def global_problem(path,
                   out_key,
                   max_scale,
                   config_path,
                   tmp_folder):
    t0 = time.time()
    agglomerator_key = 'multicut_kl'
    agglomerator = AGGLOMERATORS[agglomerator_key]
    last_scale = max_scale + 1

    with open(config_path) as f:
        n_threads = json.load(f)['n_threads']

    f_graph = z5py.File(os.path.join(tmp_folder, 'merged_graph.n5/s%i' % last_scale))
    n_nodes = f_graph.attrs['numberOfNodes']
    ds_uvs = f_graph['edges']
    ds_uvs.n_threads = n_threads
    uv_ids = ds_uvs[:]
    ds_node_labels = f_graph['nodeLabeling']
    ds_node_labels.n_threads = n_threads
    initial_node_labeling = ds_node_labels[:]
    n_edges = len(uv_ids)

    # get the costs
    ds_costs = f_graph['costs']
    ds_costs.n_threads = n_threads
    costs = ds_costs[:]
    assert len(costs) == n_edges, "%i, %i" (len(costs), n_edges)

    # build the reduce graph and solve the global reduced problem
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    node_labeling = agglomerator(graph, costs)

    # get the labeling of initial nodes
    new_initial_node_labeling = node_labeling[initial_node_labeling]

    f_out = z5py.File(path, use_zarr_format=False)
    node_shape = (len(new_initial_node_labeling), )
    chunks = (min(len(new_initial_node_labeling), 524288), )
    ds_nodes = f_out.require_dataset(out_key, dtype='uint64',
                                     shape=node_shape, chunks=chunks)
    ds_nodes.n_threads = n_threads
    ds_nodes[:] = new_initial_node_labeling

    res_path = os.path.join(tmp_folder, 'global_problem.log')
    with open(res_path, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("max_scale", type=int)
    parser.add_argument("config_path", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()

    global_problem(args.path,
                   args.out_key,
                   args.max_scale,
                   args.config_path,
                   args.tmp_folder)
