#! /usr/bin/python

import os
import time
import argparse
import json
import subprocess
from concurrent import futures

import z5py
import luigi
import nifty
import nifty.distributed as ndist


class InitialSubgraphTask(luigi.Task):
    """
    Compute initial sub-graphs
    """

    path = luigi.Parameter()
    ws_key = luigi.Parameter()
    out_path = luigi.Parameter()
    max_jobs = luigi.Parameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def _prepare_jobs(self, n_jobs, block_list, block_shape):
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_shape': block_shape,
                          'block_list': block_jobs}
            config_path = os.path.join(self.tmp_folder, 'initial_subgraph_config_job%i.json' % job_id)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id):
        script_path = os.path.join(self.tmp_folder, 'initial_subgraph.py')
        config_path = os.path.join(self.tmp_folder, 'initial_subgraph_config_job%i.json' % job_id)
        command = '%s %s %s %s %i %s %s' % (script_path, self.path, self.ws_key, self.out_path,
                                            job_id, self.tmp_folder, config_path)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_initial_subgraph_%i' % job_id)
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_initial_subgraph_%i.err' % job_id)
        bsub_command = 'bsub -J initial_subgraph_%i -We %i -o %s -e %s \'%s\'' % (job_id,
                                                                                  self.time_estimate,
                                                                                  log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _collect_outputs(self, block_list):
        times = []
        processed_blocks = []
        for block_id in block_list:
            res_file = os.path.join(self.tmp_folder, 'initial_subgraph_block%i.json' % block_id)
            try:
                with open(res_file) as f:
                    res = json.load(f)
                times.append(res['t'])
                processed_blocks.append(block_id)
                os.remove(res_file)
            except Exception:
                continue
        return processed_blocks, times

    def run(self):
        from production import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'initial_subgraph.py'),
                              os.path.join(self.tmp_folder, 'initial_subgraph.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            roi = config.get('roi', None)

        # get the shape and blocking
        ws = z5py.File(self.path)[self.ws_key]
        shape = ws.shape
        f_graph = z5py.File(self.out_path, use_zarr_format=False)
        f_graph.attrs['shape'] = shape

        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        # check if we have a ROI and adapt the block list if we do
        if roi is None:
            n_blocks = blocking.numberOfBlocks
            block_list = list(range(n_blocks))
        else:
            block_list = blocking.getBlockIdsOverlappingBoundingBox(roi[0],
                                                                    roi[1],
                                                                    [0, 0, 0]).tolist()
            n_blocks = len(block_list)

        # find the actual number of jobs and prepare job configs
        n_jobs = min(n_blocks, self.max_jobs)
        self._prepare_jobs(n_jobs, block_list, block_shape)

        # submit the jobs
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(n_jobs) as tp:
                tasks = [tp.submit(self._submit_job, job_id) for job_id in range(n_jobs)]
                [t.result() for t in tasks]
        else:
            for job_id in range(n_jobs):
                self._submit_job(job_id)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        # check the job outputs
        processed_blocks, times = self._collect_outputs(block_list)
        assert len(processed_blocks) == len(times)
        success = len(processed_blocks) == n_blocks

        # write output file if we succeed, otherwise write partial
        # success to different file and raise exception
        if success:
            out = self.output()
            # TODO does 'out' support with block?
            fres = out.open('w')
            json.dump({'times': times}, fres)
            fres.close()
        else:
            log_path = os.path.join(self.tmp_folder, 'initial_subgraph_partial.json')
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("InitialSubgraphTask failed, %i / %i blocks processed, " % (len(processed_blocks),
                                                                                           n_blocks) +
                               "serialized partial results to %s" % log_path)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'initial_subgraph.log'))


def extract_subgraph_from_roi(block_id, blocking, labels_path, labels_key, graph_path):
    print("Processign block", block_id)
    t0 = time.time()
    halo = [1, 1, 1]
    block = blocking.getBlockWithHalo(block_id, halo)
    outer_block, inner_block = block.outerBlock, block.innerBlock
    # we only need the halo into one direction,
    # hence we use the outer-block only for the end coordinate
    begin = inner_block.begin
    end = outer_block.end

    block_key = 'sub_graphs/s0/block_%i' % block_id
    ndist.computeMergeableRegionGraph(labels_path, labels_key,
                                      begin, end,
                                      graph_path, block_key)
    print("done", block_id)
    return time.time() - t0


def initial_subgraph_extraction(labels_path, labels_key, graph_path,
                                job_id, tmp_folder, config_path):
    labels = z5py.File(labels_path)[labels_key]
    shape = labels.shape

    with open(config_path) as f:
        config = json.load(f)
        block_list = config['block_list']
        block_shape = config['block_shape']

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))

    for block_id in block_list:
        t0 = extract_subgraph_from_roi(block_id, blocking, labels_path, labels_key, graph_path)
        res_file = os.path.join(tmp_folder, 'initial_subgraph_block%i.json' % block_id)
        with open(res_file, 'w') as f:
            json.dump({'t': t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)
    parser.add_argument("graph_path", type=str)
    parser.add_argument("job_id", type=int)
    parser.add_argument("tmp_folder", type=str)
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    initial_subgraph_extraction(args.labels_path, args.labels_key,
                                args.graph_path, args.job_id,
                                args.tmp_folder, args.config_path)
