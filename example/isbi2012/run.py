#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi
import h5py
import z5py

from cluster_tools import MulticutSegmentationWorkflow
from cluster_tools.multicut import SubSolutionsWorkflow


def run(shebang, with_rf=False):
    input_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'
    example_path = './isbi_exp.n5'
    input_key = 'volumes/affinties'

    max_jobs = 8
    configs = MulticutSegmentationWorkflow.get_config()

    global_conf = configs['global']
    global_conf.update({'shebang': shebang,
                        'block_shape': (25, 256, 256)})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_conf, f)

    ws_conf = configs['watershed']
    ws_conf.update({'sigma_weights': 0, 'channel_begin': 1, 'channel_end': 3})
    with open('./configs/watershed.config', 'w') as f:
        json.dump(ws_conf, f)

    if with_rf:
        feat_config = configs['block_edge_features']
        feat_config.update({'filters': ['gaussianSmoothing', 'laplacianOfGaussian'],
                            'sigmas': [1., 2., 4.], 'apply_in_2d': True})
        rf_path = './rf.pkl'
    else:
        feat_config = configs['block_edge_features']
        feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]})
        rf_path = ''

    with open('./configs/block_edge_features.config', 'w') as f:
        json.dump(feat_config, f)

    ret = luigi.build([MulticutSegmentationWorkflow(input_path=input_path,
                                                    input_key='volumes/affinities',
                                                    ws_path=example_path,
                                                    ws_key='volumes/watersheds',
                                                    problem_path=example_path,
                                                    node_labels_path=example_path,
                                                    node_labels_key='node_labels',
                                                    output_path=example_path,
                                                    output_key='volumes/segmentation',
                                                    rf_path=rf_path,
                                                    n_scales=1,
                                                    config_dir='./configs',
                                                    tmp_folder='./tmp',
                                                    target='local',
                                                    max_jobs=max_jobs)], local_scheduler=True)
    if ret:
        from cremi_tools.viewer.volumina import view
        with z5py.File(input_path) as f:
            affs = f['volumes/affinities'][:3].transpose((1, 2, 3, 0))
        with z5py.File(example_path) as f:
            ws = f['volumes/watersheds'][:]
            data = [affs, ws]
            if 'volumes/segmentation' in f:
                seg = f['volumes/segmentation'][:]
                data.append(seg)
        view(data)


def subsolutions(shebang):
    input_path = '/home/cpape/Work/data/isbi2012/isbi2012_train_volume.h5'
    aff_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'
    example_path = './isbi_exp.n5'

    max_jobs = 8
    configs = SubSolutionsWorkflow.get_config()

    global_conf = configs['global']
    global_conf.update({'shebang': shebang,
                        'block_shape': (25, 256, 256)})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_conf, f)

    ret = luigi.build([SubSolutionsWorkflow(ws_path=example_path,
                                            ws_key='volumes/watersheds',
                                            problem_path=example_path,
                                            output_path=example_path,
                                            output_key='volumes/sub_results',
                                            n_scales=0,
                                            config_dir='./configs',
                                            tmp_folder='./tmp',
                                            target='local',
                                            max_jobs=max_jobs)], local_scheduler=True)
    if ret:
        from cremi_tools.viewer.volumina import view
        with h5py.File(input_path) as f:
            raw = f['volumes/raw'][:]
        with z5py.File(aff_path) as f:
            affs = f['volumes/affinities'][:3].transpose((1, 2, 3, 0))
        with z5py.File(example_path) as f:
            ws = f['volumes/watersheds'][:]
            seg = f['volumes/segmentation'][:]
            subseg = f['volumes/sub_results'][:]
        data = [raw, affs, ws, subseg, seg]
        labels = ['raw', 'affs', 'ws', 'sub-segmentations', 'segmentation']
        view(data, labels)


if __name__ == '__main__':
    shebang = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
    run(shebang, with_rf=False)
    # subsolutions(shebang)
