#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster-new/bin/python

import os
import json

import numpy as np
import luigi
import z5py
import skeletor.io

from cluster_tools.skeletons import SkeletonWorkflow
from cremi_tools.viewer.volumina import view


def skeletons(sample, max_jobs, target):
    """ Skeletonize cremi segmentation.

    You can obtain the data used for this examle from
    https://drive.google.com/open?id=1E6j77gV0iwquSxd7KmmuXghgFcyuP7WW
    """

    path = '/g/kreshuk/data/cremi/example/sample%s.n5' % sample
    input_key = 'segmentation/multicut'
    output_key = 'skeletons'

    config_dir = './configs'
    tmp_folder = './tmp_skeletons_%s' % sample
    os.makedirs(config_dir, exist_ok=True)

    config = SkeletonWorkflow.get_config()
    global_config = config['global']
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster-new/bin/python'
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    resolution = [40, 4, 4]
    size_threshold = 2500

    task = SkeletonWorkflow(tmp_folder=tmp_folder, config_dir=config_dir,
                            max_jobs=max_jobs, target=target,
                            input_path=path, input_key=input_key,
                            output_path=path, output_key=output_key,
                            resolution=resolution, size_threshold=size_threshold)
    success = luigi.build([task], local_scheduler=True)
    assert success


def view_skeletons(sample):
    path = '/g/kreshuk/data/cremi/example/sample%s.n5' % sample
    input_key = 'segmentation/multicut'
    output_key = 'skeletons'

    f = z5py.File(path)

    raw_key = 'raw/s0'
    ds = f[raw_key]
    ds.n_threads = 8
    raw = ds[:]

    ds = f[input_key]
    ds.n_threads = 8
    seg = ds[:]

    skel_vol = np.zeros_like(seg, dtype='uint32')
    ds_skels = f[output_key]
    seg_ids = np.unique(seg)
    for seg_id in seg_ids:
        nodes, _ = skeletor.io.read_n5(ds_skels, seg_id)
        if nodes is None:
            continue

        coords = tuple(np.array([n[i] for n in nodes])
                       for i in range(3))
        skel_vol[coords] = seg_id

    view([raw, seg, skel_vol])


if __name__ == '__main__':
    sample = 'A'
    skeletons(sample, 32, 'local')
    view_skeletons(sample)
