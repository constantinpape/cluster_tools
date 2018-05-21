#! /usr/bin/python

import os
import time
import argparse
import json
import pickle
import subprocess

# import vigra
import numpy as np
import nifty
import nifty.graph.rag as nrag
import nifty.graph.lifted_mulitcut as nlmc
import z5py
import cremi_tools.segmentation as cseg
import luigi
from concurrent import futures
# TODO needs to be in pythonpath
from production import features as feat


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

    def _prepare_jobs(self, n_jobs, n_blocks, config, prefix):
        block_list = list(range(n_blocks))
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
                have_roi = True

        # find the shape and number of blocks
        f5 = z5py.File(self.path)
        shape = f5[self.ws_key].shape

        # make the output dataset
        f5.require_dataset(self.out_key, shape=shape,
                           chunks=chunks, dtype='uint64', compression='gzip')

        # prepare the jobs for the first (not shifted) blocking
        blocking1 = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks1 = blocking1.numberOfBlocks
        n_jobs1 = min(n_blocks1, self.max_jobs)
        config.update({'block_shift': None})
        self._prepare_jobs(n_jobs1, n_blocks1, config, 'a')

        blocking2 = nifty.tools.blocking([0, 0, 0], shape, block_shape,
                                         blockShift=block_shift)
        n_blocks2 = blocking2.numberOfBlocks
        n_jobs2 = min(n_blocks2, self.max_jobs)
        config.update({'block_shift': block_shift})
        self._prepare_jobs(n_jobs2, n_blocks2, config, 'b')

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
            msg = "FillingWatershedTask failed "
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


def compute_mc(ws, affs, offsets, n_labels):
    # compute the region adjacency graph
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=1)
    uv_ids = rag.uvIds()

    # compute the features and get edge probabilities (from mean affinities)
    # and edge sizes
    features = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets,
                                                       numberOfThreads=1)
    probs = features[:, 0]
    sizes = features[:, -1].astype('uint64')

    # compute multicut
    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    # transform probabilities to costs
    costs = mc.probabilities_to_costs(probs)
    # set edges connecting to 0 (= ignore label) to repulsive
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = -100
    # solve the mc problem
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)

    # get indicators for merge !
    # and return uv-ids, edge indicators and edge sizes
    edge_indicator = (node_labels[uv_ids[:, 0]] == node_labels[uv_ids[:, 1]]).astype('uint8')

    return uv_ids, edge_indicator, sizes


def compute_lmc(ws, affs, glia, offsets, n_labels, lifted_rf, lifted_nh):
    # compute the region adjacency graph
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=1)
    uv_ids = rag.uvIds()

    # compute the features and get edge probabilities (from mean affinities)
    # and edge sizes
    features = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets,
                                                       numberOfThreads=1)
    local_probs = features[:, 0]
    sizes = features[:, -1].astype('uint64')

    # remove all edges connecting to the ignore label, because
    # they introduce short-cut lifted edges
    valid_edges = (uv_ids != 0).all(axis=1)
    uv_ids = uv_ids[valid_edges]
    local_probs = local_probs[valid_edges]
    sizes = sizes[valid_edges]

    # build the original graph and lifted objective
    # with lifted uv-ids
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    lifted_objective = nlmc.liftedMulticutObjective(graph)
    lifted_objective.insertLiftedEdgesBfs(lifted_nh)
    # TODO is the sort necessary?
    # lifted_uv_ids = np.sort(lifted_graph.liftedUvIds(), axis=1)
    lifted_uv_ids = lifted_objective.liftedUvIds()

    # get features for the lifted edges
    lifted_feats = np.concatenate([feat.region_features(ws, lifted_uv_ids, glia),
                                   feat.ucm_features(n_labels, uv_ids, lifted_uv_ids, local_probs),
                                   feat.clustering_features(graph, local_probs, lifted_uv_ids)], axis=1)
    lifted_probs = lifted_rf.predict_proba(lifted_feats)[:, 1]

    # turn probabilities into costs
    local_costs = cseg.transform_probabilities_to_costs(local_probs)
    lifted_costs = cseg.transform_probabilities_to_costs(lifted_probs)
    # weight the costs
    n_local, n_lifted = len(uv_ids), len(lifted_uv_ids)
    total = float(n_lifted) + n_local
    # TODO does this make sense ?
    local_costs *= (n_lifted / total)
    lifted_costs *= (n_local / total)

    # update the lmc objective
    lifted_objective.setCosts(uv_ids, local_costs)
    lifted_objective.setCosts(lifted_uv_ids, lifted_costs)

    # compute lifted multicut
    solver_ga = lifted_objective.liftedMulticutGreedyAdditiveFactory().create(lifted_objective)
    node_labels = solver_ga.optimize()
    solver_kl = lifted_objective.liftedMulticutKernighanLinFactory().create(lifted_objective)
    node_labels = solver_kl.optimize(node_labels)

    # get indicators for merge !
    # and return uv-ids, edge indicators and edge sizes
    edge_indicator = (node_labels[uv_ids[:, 0]] == node_labels[uv_ids[:, 1]]).astype('uint8')
    return uv_ids, edge_indicator, sizes


def process_block(ds_ws, ds_affs, blocking, block_id, offsets,
                  lifted_rf=None, lifted_nh=None):

    print("Process block", block_id)
    # load the segmentation
    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    ws = ds_ws[bb]

    # if this block only contains a single segment id (usually 0 = ignore label) continue
    ws_ids = np.unique(ws)
    if len(ws_ids) == 1:
        return None
    n_labels = int(ws_ids[-1]) + 1

    # TODO should we do this ?
    # map to a consecutive segmentation to speed up graph computations
    # ws, max_id, mapping = vigra.analysis.relabelConsecutive(ws, keep_zeros=True, start_label=1)

    # load the affinities
    n_channels = len(offsets)
    bb_affs = (slice(0, n_channels),) + bb
    affs = ds_affs[bb_affs]
    # convert affinities to float and invert them
    # to get boundary probabilities
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = 1. - affs

    if lifted_rf is None:
        uv_ids, edge_indicator, sizes = compute_mc(ws, affs, offsets, n_labels)
    else:
        assert lifted_nh is not None
        n_aff_chans = ds_affs.shapa[0]
        bb_glia = (slice(n_aff_chans - 1, n_aff_chans),) + bb
        glia = ds_affs[bb_glia]
        uv_ids, edge_indicator, sizes = compute_lmc(ws, affs, glia, offsets, n_labels,
                                                    lifted_rf, lifted_nh)

    return uv_ids, edge_indicator, sizes


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
    weight_edges = config['weight_edges']
    lifted_rf_path = config.get('lifted_rf_path', None)
    lifted_nh = config.get('lifted_nh', None)

    # open all n5 datasets
    ds_ws = z5py.File(path)[ws_key]
    ds_affs = z5py.File(path)[aff_key]
    shape = ds_ws.shape

    blocking = nifty.tools.blocking([0, 0, 0], list(shape), block_shape,
                                    blockShift=block_shift)

    if lifted_rf_path is not None:
        with open(lifted_rf_path, 'rb') as f:
            lifted_rf = pickle.load(f)
    else:
        lifted_rf = None

    results = [process_block(ds_ws, ds_affs,
                             blocking, block_id, offsets,
                             lifted_rf, lifted_nh)
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
                                                                weightEdges=weight_edges)

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
