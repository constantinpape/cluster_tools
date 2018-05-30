#! /usr/bin/python

import os
import time
import argparse
import subprocess
import json
from concurrent import futures

import z5py
import nifty
import nifty.distributed as ndist
import luigi


class MergeSubgraphScalesTask(luigi.Task):
    """
    Merge subgraphs on scale level
    """

    path = luigi.Parameter()
    ws_key = luigi.Parameter()
    out_path = luigi.Parameter()
    scale = luigi.IntParameter()
    max_jobs = luigi.IntParameter()
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
            config_path = os.path.join(self.tmp_folder, 'graph_scale%i_config_job%i.json' % (self.scale, job_id))
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id):
        script_path = os.path.join(self.tmp_folder, 'merge_graph_scales.py')
        config_path = os.path.join(self.tmp_folder, 'graph_scale%i_config_job%i.json' % (self.scale, job_id))
        command = '%s %s %i %i %s %s' % (script_path, self.out_path, self.scale, job_id,
                                         config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_graph%i_scale_%i' % (self.scale, job_id))
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_graph_scale%i_%i.err' % (self.scale, job_id))
        bsub_command = 'bsub -J graph_scale_%i -We %i -o %s -e %s \'%s\'' % (job_id,
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
            res_file = os.path.join(self.tmp_folder, 'graph_scale%i_block%i.json' % (self.scale, block_id))
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
        util.copy_and_replace(os.path.join(file_dir, 'merge_graph_scales.py'),
                              os.path.join(self.tmp_folder, 'merge_graph_scales.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            init_block_shape = config['block_shape']
            roi = config.get('roi', None)

        block_shape = [bs * 2**self.scale for bs in init_block_shape]

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
        self._prepare_jobs(n_jobs, block_list, init_block_shape)

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
            log_path = os.path.join(self.tmp_folder, 'merge_graph_scale%i_partial.json' % self.scale)
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("MergeGraphScalesTask failed, %i / %i blocks processed, " % (len(processed_blocks),
                                                                                            n_blocks) +
                               "serialized partial results to %s" % log_path)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'merge_graph_scale%i.log' % self.scale))


def merge_subblocks(block_id, blocking, previous_blocking, graph_path, scale):
    t0 = time.time()
    block = blocking.getBlock(block_id)
    input_key = 'sub_graphs/s%i/block_' % (scale - 1,)
    output_key = 'sub_graphs/s%i/block_%i' % (scale, block_id)
    block_list = previous_blocking.getBlockIdsInBoundingBox(roiBegin=block.begin,
                                                            roiEnd=block.end,
                                                            blockHalo=[0, 0, 0])
    ndist.mergeSubgraphs(graph_path,
                         blockPrefix=input_key,
                         blockIds=block_list.tolist(),
                         outKey=output_key)
    return time.time() - t0


def merge_graph_scale(graph_path, scale, job_id, config_path, tmp_folder):

    with open(config_path) as f:
        config = json.load(f)
        initial_block_shape = config['block_shape']
        block_list = config['block_list']

    factor = 2**scale
    previous_factor = 2**(scale - 1)
    block_shape = [factor * bs for bs in initial_block_shape]
    previous_block_shape = [previous_factor * bs for bs in initial_block_shape]

    f_graph = z5py.File(graph_path)
    shape = f_graph.attrs['shape']
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    previous_blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                             roiEnd=list(shape),
                                             blockShape=previous_block_shape)
    for block_id in block_list:
        t0 = merge_subblocks(block_id, blocking, previous_blocking, graph_path, scale)
        res_file = os.path.join(tmp_folder, 'graph_scale%i_block%i.json' % (scale, block_id))
        with open(res_file, 'w') as f:
            json.dump({"t": t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("job_id", type=int)
    parser.add_argument("config_path", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()

    merge_graph_scale(args.graph_path, args.scale, args.job_id,
                      args.config_path, args.tmp_folder)
