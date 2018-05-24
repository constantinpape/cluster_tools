#! /usr/bin/python

import os
import time
import argparse
import json
import pickle
import subprocess

import numpy as np
import z5py
import luigi
from concurrent import futures
from production import multicut

import nifty

# import vigra
# import nifty.graph.rag as nrag


# TODO more clean up (job config files)
# TODO computation with rois
class MergeVotesTask(luigi.Task):
    """
    Run all thresholding tasks
    """

    # path to the n5 file and keys
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    ws_key = luigi.Parameter()
    out_key = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    # path to the configuration
    # TODO allow individual paths for individual blocks
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)
    # TODO optional parameter to just run a subset of blocks

    def requires(self):
        return self.dependency

    def _prepare_jobs(self, n_jobs, block_list, config, prefix):
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'config': config,
                          'block_list': block_jobs}
            config_path = os.path.join(self.tmp_folder, 'merge_votes_config_%s_job%i.json' % (prefix, job_id))
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id, prefix):
        script_path = os.path.join(self.tmp_folder, 'compute_merge_votes.py')
        assert os.path.exists(script_path)
        config_path = os.path.join(self.tmp_folder, 'merge_votes_config_%s_job%i.json' % (prefix,
                                                                                          job_id))
        command = '%s %s %s %s %i %s %s %s' % (script_path, self.path, self.aff_key, self.ws_key,
                                               job_id, self.tmp_folder, config_path, prefix)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_merge_votes_%s_%i' % (prefix, job_id))
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_merge_votes_%s_%i.err' % (prefix, job_id))
        bsub_command = 'bsub -J merge_votes_%i -We %i -o %s -e %s \'%s\'' % (job_id,
                                                                             self.time_estimate,
                                                                             log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _submit_jobs(self, n_jobs, prefix):
        from .. import util
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(n_jobs) as tp:
                tasks = [tp.submit(self._submit_job, job_id, prefix)
                         for job_id in range(n_jobs)]
                [t.result() for t in tasks]
        else:
            for job_id in range(n_jobs):
                self._submit_job(job_id, prefix)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

    def _collect_outputs(self, n_jobs, prefix):
        times = []
        processed_jobs = []
        for job_id in range(n_jobs):
            save_path1 = os.path.join(self.tmp_folder, 'compute_merge_votes_%s_%i.npy' % (prefix, job_id))
            save_path2 = os.path.join(self.tmp_folder, 'compute_merge_uvs_%s_%i.npy' % (prefix, job_id))
            out_path = os.path.join(self.tmp_folder, 'compute_merge_times_%s_%i.json' % (prefix, job_id))
            try:
                assert os.path.exists(save_path1) and os.path.exists(save_path2)
                with open(out_path) as f:
                    res = json.load(f)
                    times.append(res['t'])
                processed_jobs.append(job_id)
                os.remove(out_path)
            except Exception:
                continue
        return processed_jobs, times

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'compute_merge_votes.py'),
                              os.path.join(self.tmp_folder, 'compute_merge_votes.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape2']
            chunks = tuple(config['chunks'])
            # we need to pop the block shift from the config
            # because the first blocking is without block shift !
            block_shift = config.pop('block_shift')
            # TODO support computation with roi
            if 'roi' in config:
                roi = config['roi']
            else:
                roi = None

        # find the shape and number of blocks
        f5 = z5py.File(self.path)
        shape = f5[self.ws_key].shape

        # make the output dataset
        f5.require_dataset(self.out_key, shape=shape,
                           chunks=chunks, dtype='uint64', compression='gzip')

        # for debugging
        f5.require_dataset('segmentation/debug_a', shape=shape,
                           chunks=chunks, dtype='uint64', compression='gzip')
        f5.require_dataset('segmentation/debug_b', shape=shape,
                           chunks=chunks, dtype='uint64', compression='gzip')

        # prepare the jobs for the first (not shifted) blocking
        blocking1 = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks1 = blocking1.numberOfBlocks
        config.update({'block_shift': None})
        if roi is None:
            block_list1 = list(range(n_blocks1))
        else:
            block_list1 = blocking1.getBlockIdsOverlappingBoundingBox(roi[0], roi[1], [0, 0, 0]).tolist()
        n_jobs1 = min(len(block_list1), self.max_jobs)
        self._prepare_jobs(n_jobs1, block_list1, config, 'a')

        blocking2 = nifty.tools.blocking([0, 0, 0], shape, block_shape,
                                         blockShift=block_shift)
        n_blocks2 = blocking2.numberOfBlocks
        config.update({'block_shift': block_shift})
        if roi is None:
            block_list2 = list(range(n_blocks2))
        else:
            block_list2 = blocking2.getBlockIdsOverlappingBoundingBox(roi[0], roi[1], [0, 0, 0]).tolist()
        n_jobs2 = min(len(block_list2), self.max_jobs)
        self._prepare_jobs(n_jobs2, block_list2, config, 'b')

        print("Start blocks a")
        self._submit_jobs(n_jobs1, 'a')
        print("Start blocks b")
        self._submit_jobs(n_jobs2, 'b')

        processed_jobs_a, times_a = self._collect_outputs(n_jobs1, 'a')
        success_a = len(processed_jobs_a) == n_jobs1

        processed_jobs_b, times_b = self._collect_outputs(n_jobs2, 'b')
        success_b = len(processed_jobs_b) == n_jobs2

        success = success_a and success_b

        if success:
            out = self.output()
            # TODO does 'out' support with block?
            fres = out.open('w')
            json.dump({'times_a': times_a, 'times_b': times_b}, fres)
            fres.close()

        else:
            log_path = os.path.join(self.tmp_folder, 'compute_merge_votes_partial.json')
            msg = "MergeVotesTask failed "
            msg += "%i / %i jobs processed for blocking a " % (len(processed_jobs_a), n_jobs1)
            msg += "%i / %i jobs processed for blocking b " % (len(processed_jobs_b), n_jobs2)
            msg += "serialized partial results to %s" % log_path
            with open(log_path, 'w') as out:
                json.dump({'times_a': times_a, "times_b": times_b,
                           'processed_jobs_a': processed_jobs_a,
                           'processed_jobs_b': processed_jobs_b}, out)
            raise RuntimeError(msg)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'compute_merge_votes.log'))


def process_block(ds_ws, ds_affs, blocking, block_id, offsets,
                  use_lifted=False, rf=None, lifted_nh=None,
                  weight_mulitcut_edges=False,
                  weighting_exponent=1, ds_out=None):

    print("Process block", block_id)
    # load the segmentation
    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    ws = ds_ws[bb]

    # FIXME this causes false merges and is broken
    # we map to a consecutive segmentation to speed up graph computations
    # ws, max_id, mapping = vigra.analysis.relabelConsecutive(ws, keep_zeros=True, start_label=1)
    # # if this block only contains a single element, return (usually 0 = ignore label) continue
    # if len(mapping) == 1:
    #     return None
    # ws = ws.astype('uint32')
    # reverse_mapping = {val: key for key, val in mapping.items()}
    # n_labels = int(max_id) + 1

    ws = ws.astype('uint32')
    n_labels = int(ws.max()) + 1

    # load the affinities
    n_channels = len(offsets)
    bb_affs = (slice(0, n_channels),) + bb
    affs = ds_affs[bb_affs]
    # convert affinities to float and invert them
    # to get boundary probabilities
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = 1. - affs

    if use_lifted:
        if rf is None:
            uv_ids, merge_indicator, sizes = multicut.compute_lmc(ws, affs, offsets, n_labels,
                                                                  weight_mulitcut_edges,
                                                                  weighting_exponent)

        else:
            n_aff_chans = ds_affs.shape[0]
            bb_glia = (slice(n_aff_chans - 1, n_aff_chans),) + bb
            glia = ds_affs[bb_glia]
            if glia.dtype == np.dtype('uint8'):
                glia = glia.astype('float32') / 255.
            uv_ids, merge_indicator, sizes = multicut.compute_lmc_learned(ws, affs, glia,
                                                                          offsets, n_labels,
                                                                          rf, lifted_nh,
                                                                          weight_mulitcut_edges,
                                                                          weighting_exponent)
    else:
        if rf is None:
            uv_ids, merge_indicator, sizes = multicut.compute_mc(ws, affs,
                                                                 offsets, n_labels,
                                                                 weight_mulitcut_edges,
                                                                 weighting_exponent)
        else:
            uv_ids, merge_indicator, sizes = multicut.compute_mc_learned(ws, affs,
                                                                         offsets, n_labels,
                                                                         weight_mulitcut_edges,
                                                                         weighting_exponent, rf)
    # check for empty results
    if uv_ids is None:
        return None

    # for debugging
    # if ds_out is not None:
    #     rag = nrag.gridRag(ws, numberOfLabels=n_labels, numberOfThreads=1)
    #     ufd = nifty.ufd.ufd(n_labels)
    #     uv_ids = rag.uvIds()
    #     merge_pairs = uv_ids[merge_indicator.astype('bool')]
    #     ufd.merge(merge_pairs)
    #     node_labels = ufd.elementLabeling()
    #     seg = nrag.projectScalarNodeDataToPixels(rag, node_labels)
    #     ds_out[bb] = seg

    # FIXME this causes false merges and is broken
    # map back to the original ids
    # uv_ids = nifty.tools.takeDict(reverse_mapping, uv_ids).astype('uint64')
    return uv_ids, merge_indicator, sizes


def compute_merge_votes(path, aff_key, ws_key, job_id,
                        tmp_folder, config_path, prefix):

    t0 = time.time()
    # load the blocks to be processed and the configuration from the input config file
    with open(config_path) as f:
        input_config = json.load(f)
    block_list = input_config['block_list']
    config = input_config['config']
    block_shape, block_shift = config['block_shape2'], config['block_shift']
    offsets = config['affinity_offsets']
    weight_merge_edges = config['weight_merge_edges']
    weight_mulitcut_edges = config['weight_multicut_edges']
    weighting_exponent = config.get('weighting_exponent', 1.)
    use_lifted = config.get('use_lifted', False)
    rf_path = config.get('rf_path', None)
    lifted_nh = config.get('lifted_nh', None)

    # open all n5 datasets
    ds_ws = z5py.File(path)[ws_key]
    ds_affs = z5py.File(path)[aff_key]
    shape = ds_ws.shape

    # # for debugging
    # ds_out = z5py.File(path)['segmentation/debug_' + prefix]

    blocking = nifty.tools.blocking([0, 0, 0], list(shape), block_shape,
                                    blockShift=block_shift)

    if rf_path is not None:
        if use_lifted:
            assert lifted_nh is not None
            assert len(rf_path) == 1
            with open(rf_path[0], 'rb') as f:
                rf = pickle.load(f)
        else:
            assert len(rf_path) == 2
            rf = []
            with open(rf_path[0], 'rb') as f:
                rf.append(pickle.load(f))
            with open(rf_path[1], 'rb') as f:
                rf.append(pickle.load(f))
    else:
        rf = None

    results = [process_block(ds_ws, ds_affs,
                             blocking, block_id,
                             offsets,
                             use_lifted, rf, lifted_nh,
                             weight_mulitcut_edges, weighting_exponent)  # ds_out=ds_out)
               for block_id in block_list]
    results = [res for res in results if res is not None]

    # all blocks could be empty
    if not results:
        merged_uvs, merge_votes = [], []

    else:
        uv_ids = np.concatenate([res[0] for res in results], axis=0)
        indicators = np.concatenate([res[1] for res in results], axis=0)
        sizes = np.concatenate([res[2] for res in results], axis=0)

        # compute nominator and denominator of merge votes
        merged_uvs, merge_votes = nifty.tools.computeMergeVotes(uv_ids, indicators, sizes,
                                                                weightEdges=weight_merge_edges)

        # # debugging
        # if ds_out is not None:
        #     vote_ratio = merge_votes[:, 0].astype('float64') / merge_votes[:, 1]

        #     merge_threshold = config['merge_threshold']
        #     # merge all node pairs whose ratio is above the merge threshold
        #     merges = vote_ratio > merge_threshold
        #     n_labels = int(merged_uvs.max()) + 1
        #     merge_node_pairs = merged_uvs[merges]

        #     ufd = nifty.ufd.ufd(n_labels)

        #     ufd.merge(merge_node_pairs)
        #     node_labels = ufd.elementLabeling()
        #     for block_id in block_list:
        #         block = blocking.getBlock(block_id)
        #         bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        #         ws = ds_ws[bb]
        #         seg = nifty.tools.take(node_labels, ws)
        #         try:
        #             ds_out[bb] = seg
        #         except RuntimeError:
        #             pass

    # TODO should we also serialize the block-level results for better fault tolerance ?!
    # serialize the job level results
    save_path1 = os.path.join(tmp_folder, 'compute_merge_votes_%s_%i.npy' % (prefix, job_id))
    np.save(save_path1, merge_votes)
    save_path2 = os.path.join(tmp_folder, 'compute_merge_uvs_%s_%i.npy' % (prefix, job_id))
    np.save(save_path2, merged_uvs)
    out_path = os.path.join(tmp_folder, 'compute_merge_times_%s_%i.json' % (prefix, job_id))
    with open(out_path, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('aff_key', type=str)
    parser.add_argument('ws_key', type=str)
    parser.add_argument('job_id', type=int)
    parser.add_argument('tmp_folder', type=str)
    parser.add_argument('config_path', type=str)
    parser.add_argument('prefix', type=str)

    args = parser.parse_args()
    compute_merge_votes(args.path,
                        args.aff_key,
                        args.ws_key,
                        args.job_id,
                        args.tmp_folder,
                        args.config_path,
                        args.prefix)
