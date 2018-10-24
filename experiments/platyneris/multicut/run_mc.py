#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import numpy as np
import vigra
import luigi
import h5py
import z5py

import nifty.distributed as ndist
import nifty.ufd as nufd
import nifty.tools as nt

from cluster_tools import MulticutSegmentationWorkflow
from cluster_tools.utils.volume_utils import InterpolatedVolume
from cluster_tools.utils.segmentation_utils import multicut_gaec


def run_wf(block_id, tmp_folder, max_jobs,
           target='local', with_rf=False):

    input_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/predictions/val_block_0%i_unet_lr_v3_bmap.n5' % block_id
    # input_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/predictions/val_block_0%i_unet_lr_v3_ds122.n5' % block_id
    input_key = 'data'

    mask_path = './test_val_mask.h5'
    mask_key = 'data'

    # mask_path = ''
    # mask_key = ''

    if with_rf:
        exp_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/segmentation/val_block_0%i_rf.n5' % block_id
        rf_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/predictions/rf_v1.pkl'

    else:
        exp_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/segmentation/val_block_0%i.n5' % block_id
        rf_path = ''

    use_decomposer = False
    configs = MulticutSegmentationWorkflow.get_config(use_decomposer)

    if not os.path.exists('config'):
        os.mkdir('config')

    roi_begin, roi_end = None, None
    # roi_begin = [50, 0, 0]
    # roi_end = [100, 2048, 2048]

    global_config = configs['global']
    global_config.update({'shebang': "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python",
                          'roi_begin': roi_begin,
                          'roi_end': roi_end})
    with open('./config/global.config', 'w') as f:
        json.dump(global_config, f)

    ws_config = configs['watershed']
    # TODO completely debug the two-pass watershed
    two_pass_ws = False
    ws_config.update({'threshold': .25, 'apply_presmooth_2d': False,
                      'sigma_weights': (1., 2., 2.), 'apply_dt_2d': False,
                      'pixel_pitch': (2, 1, 1), 'two_pass': two_pass_ws,
                      'halo': [0, 50, 50]})
    with open('./config/watershed.config', 'w') as f:
        json.dump(ws_config, f)

    subprob_config = configs['solve_subproblems']
    subprob_config.update({'weight_edges': False,
                           'threads_per_job': max_jobs})
    with open('./config/solve_subproblems.config', 'w') as f:
        json.dump(subprob_config, f)

    feat_config = configs['block_edge_features']
    if with_rf:
        feat_config.update({'filters': ['gaussianSmoothing'],
                            'sigmas': [(0.5, 1., 1.), (1., 2., 2.),
                                       (2., 4., 4.), (4., 8., 8.)],
                            'halo': (8, 16, 16),
                            'channel_agglomeration': 'mean'})

    else:
        # feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]})
        feat_config.update({'offsets': None})
        # feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
        #                                 [-2, 0, 0], [0, -4, 0], [0, 0, -4],
        #                                 [-4, 0, 0], [0, -8, 0], [0, 0, -8],
        #                                 [-12, 0, 0], [0, -24, 0], [0, 0, -24]]})
        # feat_config.update({'filters': ['gaussianSmoothing'],
        #                     'sigmas': [(2., 4., 4.)],
        #                     'halo': (8, 16, 16),
        #                     'channel_agglomeration': 'max'})
    with open('./config/block_edge_features.config', 'w') as f:
        json.dump(feat_config ,f)

    # set number of threads for sum jobs
    if use_decomposer:
        tasks = ['merge_sub_graphs', 'merge_edge_features', 'probs_to_costs',
                 'solve_subproblems', 'decompose', 'insert']
    else:
        tasks = ['merge_sub_graphs', 'merge_edge_features', 'probs_to_costs',
                 'solve_subproblems', 'reduce_problem', 'solve_global']

    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_jobs})
        with open('./config/%s.config' % tt, 'w') as f:
            json.dump(config, f)

    ret = luigi.build([MulticutSegmentationWorkflow(input_path=input_path, input_key=input_key,
                                                    mask_path=mask_path, mask_key=mask_key,
                                                    # ws_path=exp_path, ws_key='volumes/watershed',
                                                    ws_path=input_path, ws_key='watershed',
                                                    graph_path=exp_path, features_path=exp_path,
                                                    costs_path=exp_path, problem_path=exp_path,
                                                    node_labels_path=exp_path, node_labels_key='node_labels',
                                                    output_path=exp_path, output_key='volumes/segmentation',
                                                    use_decomposition_multicut=use_decomposer,
                                                    rf_path=rf_path,
                                                    n_scales=2,
                                                    config_dir='./config',
                                                    tmp_folder=tmp_folder,
                                                    target=target,
                                                    skip_ws=True,
                                                    max_jobs=max_jobs)], local_scheduler=True)
    # ret = True
    # view the results if we are local and the
    # tasks were successfull
    if ret and target == 'local':
        print("Starting viewer")
        from cremi_tools.viewer.volumina import view

        with z5py.File(input_path) as f:
            ds = f[input_key]
            ds.n_threads = max_jobs
            affs = ds[:]
            if affs.ndim == 4:
                affs = affs.transpose((1, 2, 3, 0))

            ds = f['watershed']
            ds.n_threads = max_jobs
            ws = ds[:]

        data = [affs, ws]

        with z5py.File(exp_path) as f:
            # ds = f['volumes/watershed']
            # ds.n_threads = max_jobs
            # ws = ds[:]
            # data.append(ws)
            # shape = ds.shape

            if 'volumes/segmentation' in f:
                ds = f['volumes/segmentation']
                ds.n_threads = max_jobs
                seg = ds[:]
                data.append(seg)

        # with h5py.File('./test_val_mask.h5') as f:
        #     ds_mask = f['data'][:]
        # interp_mask = InterpolatedVolume(ds_mask, shape)
        # full_mask = interp_mask[:]
        # assert full_mask.shape == shape
        # data.append(full_mask)

        view(data)


def debug_subresult(block_id=1):
    from cremi_tools.viewer.volumina import view
    path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/segmentation/val_block_01.n5'
    tmp_folder = './tmp_plat_val'
    block_prefix = os.path.join(path, 's0', 'sub_graphs', 'block_')

    graph = ndist.Graph(os.path.join(path, 'graph'))
    block_path = block_prefix + str(block_id)
    nodes = ndist.loadNodes(block_path)
    nodes = nodes[1:]
    inner_edges, outer_edges, sub_uvs = graph.extractSubgraphFromNodes(nodes)

    block_res_path = os.path.join(tmp_folder, 'subproblem_results/s0_block%i.npy' % block_id)
    res = np.load(block_res_path)

    merge_edges = np.ones(graph.numberOfEdges, dtype='bool')
    merge_edges[res] = False
    merge_edges[outer_edges] = False

    uv_ids = graph.uvIds()
    n_nodes = int(uv_ids.max()) + 1
    ufd = nufd.ufd(n_nodes)
    ufd.merge(uv_ids[merge_edges])
    node_labels = ufd.elementLabeling()

    ws = z5py.File(path)['volumes/watershed'][:]
    seg = nt.take(node_labels, ws)
    view([ws, seg])


def debug_costs():
    from cremi_tools.viewer.volumina import view
    from nifty.graph import undirectedGraph
    path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/segmentation/val_block_01.n5'

    costs = z5py.File(path)['costs'][:]
    edges = z5py.File(path)['graph/edges'][:]
    assert len(costs) == len(edges)
    print(np.mean(costs), "+-", np.std(costs))
    print(costs.min(), costs.max())

    # import matplotlib.pyplot as plt
    # n, bins, patches = plt.hist(costs, 50)
    # plt.grid(True)
    # plt.show()

    n_nodes = int(edges.max()) + 1
    graph = undirectedGraph(n_nodes)
    graph.insertEdges(edges)

    assert graph.numberOfEdges == len(costs)
    node_labels = multicut_gaec(graph, costs)

    ds = z5py.File(path)['volumes/watershed']
    ds.n_threads = 8
    ws = ds[:]
    seg = nt.take(node_labels, ws)

    bb = np.s_[25:75, 500:1624, 100:1624]

    input_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/predictions/val_block_0%i_unet_lr_v3_ds122.n5' % block_id
    with z5py.File(input_path) as f:
        ds = f['data']
        ds.n_threads = 8
        affs = ds[(slice(0, 3),) + bb]
    view([affs.transpose((1, 2, 3, 0)), ws[bb], seg[bb]])


def debug_feats():
    path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/segmentation/val_block_01.n5'
    feats = z5py.File(path)['features'][:, 0]
    print(feats.min(), feats.max())
    print(feats.mean(), '+-', feats.std())
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(feats, 50)
    plt.grid(True)
    plt.show()


def make_mask():
    block_id = 1
    input_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/predictions/val_block_0%i_unet_lr_v3_ds122.n5' % block_id
    input_key = 'data'
    with z5py.File(input_path) as f:
        shape = f['data'].shape[1:]

    ds_shape = tuple(sh // 4 for sh in shape)
    mask = np.zeros(ds_shape, dtype='uint8')
    center = tuple(sh // 2 for sh in ds_shape)
    size = tuple(3 * ce // 4 for ce in center)
    bb = (slice(None),) + tuple(slice(ce - si, ce + si) for ce, si in zip(center[1:], size[1:]))
    print(bb)
    mask[bb] = 1
    with h5py.File('./test_val_mask.h5', 'w') as f:
        f.create_dataset('data', compression='gzip', data=mask)


if __name__ == '__main__':
    with_rf = False

    if with_rf:
        tmp_folder = './tmp_plat_val_rf'
    else:
        tmp_folder = './tmp_plat_val'

    # target = 'slurm'
    # max_jobs = 32

    target = 'local'
    max_jobs = 8

    block_id = 1
    run_wf(block_id, tmp_folder, max_jobs, target=target, with_rf=with_rf)
    # debug_feats()
    # debug_costs()
