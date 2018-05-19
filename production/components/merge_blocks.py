#! /usr/bin/python

import argparse
import time
import json
import os
import subprocess
from concurrent import futures

import numpy as np
import luigi
import z5py
import nifty


# TODO more clean up (job config files)
# TODO computation with rois
class MergeTask(luigi.Task):
    """
    Run all block-merge tasks
    """

    path = luigi.Parameter()
    out_key = luigi.Parameter()
    config_path = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def _prepare_jobs(self, n_jobs, n_blocks, block_shape):
        block_list = list(range(n_blocks))
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_shape': block_shape,
                          'block_list': block_jobs}
            config_path = os.path.join(self.tmp_folder, 'merge_blocks_config_job%i.json' % job_id)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id, offset_path):
        script_path = os.path.join(self.tmp_folder, 'merge_blocks.py')
        config_path = os.path.join(self.tmp_folder, 'merge_blocks_config_job%i.json' % job_id)
        command = '%s %s %s %i %s %s %s' % (script_path, self.path, self.out_key,
                                            job_id, config_path, offset_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_merge_blocks_%i' % job_id)
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_merge_blocks_%i.err' % job_id)
        bsub_command = 'bsub -J merge_blocks_%i -We %i -o %s -e %s \'%s\'' % (job_id,
                                                                              self.time_estimate,
                                                                              log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _collect_outputs(self, n_jobs):
        times = []
        processed_blocks = []
        for job_id in range(n_jobs):
            res_file = os.path.join(self.tmp_folder, 'merge_blocks_job%ijson' % job_id)
            try:
                with open(res_file) as f:
                    res = json.load(f)
                processed_blocks.extend(list(map(int, res.keys())))
                times.extend(list(res.values()))
                os.remove(res_file)
            except Exception:
                continue
        return processed_blocks, times

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'merge_blocks.py'),
                              os.path.join(self.tmp_folder, 'merge_blocks.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            # TODO support computation with roi
            if 'roi' in config:
                have_roi = True

        # find the shape and number of blocks
        f = z5py.File(self.path)
        ds = f[self.out_key]
        shape = ds.shape
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks

        # get the block offset path from out dependency
        offset_path = self.input().path

        # find the actual number of jobs and prepare job configs
        n_jobs = min(n_blocks, self.max_jobs)
        self._prepare_jobs(n_jobs, n_blocks, block_shape)

        # submit the jobs
        # TODO would be better to wrap this into a process pool, but
        # it will be quite a pain to make everything pickleable
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(n_jobs) as tp:
                tasks = [tp.submit(self._submit_job, job_id, offset_path)
                         for job_id in range(n_jobs)]
                [t.result() for t in tasks]
        else:
            for job_id in range(n_jobs):
                self._submit_job(job_id, offset_path)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        # check the job outputs
        processed_blocks, times = self._collect_outputs(n_jobs)
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
            log_path = os.path.join(self.tmp_folder, 'merge_blocks.json')
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("ThresholdTask failed, %i / %i blocks processed, serialized partial results to %s" % (len(processed_blocks),
                                                                                                                     n_blocks,
                                                                                                                     log_path))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'merge_blocks.log'))


def merge_blocks(path, out_key, job_id, config_path, offsets_path, tmp_folder):

    # load the config
    with open(config_path) as f:
        input_config = json.load(f)
        block_shape = input_config['block_shape']
        block_ids = input_config['block_list']

    # load the block offsets and empty blocks
    with open(offsets_path) as f:
        offset_config = json.load(f)
    empty_blocks = offset_config['empty_blocks']
    offsets = offset_config['offsets']

    ds = z5py.File(path)[out_key]
    shape = ds.shape
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    # find the assignments for each block in block-ids
    node_assignments = []
    times = []
    for block_id in block_ids:
        t0 = time.time()
        # get current block
        block = blocking.getBlock(block_id)
        off = offsets[block_id]

        # iterate over the neighbors, find adjacent component
        # and merge them
        assignments = []
        to_lower = False
        for dim in range(3):

            # find the block id of the overlapping neighbor
            ngb_id = blocking.getNeighborId(block_id, dim, to_lower)
            if ngb_id == -1:
                continue
            # don't stitch with empty blocks
            if ngb_id in empty_blocks:
                continue

            # find the bb for the adjacent faces of both blocks
            adjacent_bb = tuple(slice(block.begin[i], block.end[i]) if i != dim else
                                slice(block.end[i] - 1, block.end[i] + 1)
                                for i in range(3))
            # load the adjacent faces
            adjacent_faces = ds[adjacent_bb]

            # get the face of this block and the ngb block
            bb = tuple(slice(None) if i != dim else slice(0, 1) for i in range(3))
            bb_ngb = tuple(slice(None) if i != dim else slice(1, 2) for i in range(3))
            face, face_ngb = adjacent_faces[bb], adjacent_faces[bb_ngb]

            # add the offsets
            labeled = face != 0
            face[labeled] += off
            off_ngb = offsets[ngb_id]
            face_ngb[face_ngb != 0] += off_ngb

            # find the assignments via touching (non-zero) component ids
            ids_a, ids_b = face[labeled], face_ngb[labeled]
            assignment = np.concatenate([ids_a[None], ids_b[None]], axis=0).transpose()
            if assignment.size:
                assignment = np.unique(assignment, axis=0)
                # filter zero assignments
                valid_assignment = (assignment != 0).all(axis=1)
                assignments.append(assignment[valid_assignment])

        if assignments:
            assignments = np.concatenate(assignments, axis=0)
            node_assignments.append(assignments)

        times.append(time.time() - t0)

    if node_assignments:
        node_assignments = np.concatenate(node_assignments, axis=0)

    # serialize the node assignments
    out_path = os.path.join(tmp_folder, 'node_assignments_job%i.npy' % job_id)
    np.save(out_path, node_assignments)

    # serialize the block times
    save_path = os.path.join(tmp_folder, 'merge_blocks_job%ijson' % job_id)
    with open(save_path, 'w') as f:
        json.dump({block_id: tt for block_id, tt in zip(block_ids, times)}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('job_id', type=int)
    parser.add_argument('config_path', type=str)
    parser.add_argument('offsets_path', type=str)
    parser.add_argument('tmp_folder', type=str)
    args = parser.parse_args()
    merge_blocks(args.path, args.out_key, args.job_id, args.config_path,
                 args.offsets_path, args.tmp_folder)
