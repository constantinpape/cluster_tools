#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster-new/bin/python

import os
import json
from concurrent import futures

import numpy as np
import luigi
import z5py
import skeletor.io

from cluster_tools.skeletons import SkeletonWorkflow


def check_scale(scale):
    path = '/g/kreshuk/data/FIB25/data.n5'
    input_key = 'volumes/paintera/multicut/data/s%i' % scale
    f = z5py.File(path)
    ds = f[input_key]
    shape = ds.shape
    print(shape)


def skeletonize(scale, target, max_jobs):

    path = '/g/kreshuk/data/FIB25/data.n5'
    input_key = 'volumes/paintera/multicut/data/s%i' % scale
    output_key = 'skeletons/s%i' % scale

    config_dir = './configs'
    tmp_folder = './tmp_skeletons_%i' % scale
    os.makedirs(config_dir, exist_ok=True)

    config = SkeletonWorkflow.get_config()
    global_config = config['global']
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster-new/bin/python'
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    config = config['skeletonize']
    config.update({'time_limit': 600, 'mem_limit': 16})
    with open(os.path.join(config_dir, 'skeletonize.config'), 'w') as f:
        json.dump(config, f)

    resolution = [8, 8, 8]
    size_threshold = 2000
    max_id = z5py.File(path)['volumes/paintera/multicut/data'].attrs['maxId']

    task = SkeletonWorkflow(tmp_folder=tmp_folder, config_dir=config_dir,
                            max_jobs=max_jobs, target=target,
                            input_path=path, input_key=input_key,
                            output_path=path, output_key=output_key,
                            resolution=resolution, size_threshold=size_threshold,
                            max_id=max_id)
    success = luigi.build([task], local_scheduler=True)
    assert success


def skeletons_to_volume(scale, n_threads):
    path = '/g/kreshuk/data/FIB25/data.n5'
    f = z5py.File(path)
    seg_key = 'volumes/paintera/multicut/data/s%i' % scale
    in_key = 'skeletons/s%i' % scale
    out_key = 'skeletons/volumes/s%i' % scale

    shape = f[seg_key].shape
    chunks = f[seg_key].chunks
    seg = np.zeros(shape, dtype='uint64')
    ds_in = f[in_key]

    def seg_to_vol(seg_id):
        nodes, _ = skeletor.io.read_n5(ds_in, seg_id)
        if nodes is None:
            return
        print(seg_id, '/', ds_in.shape[0])
        coords = tuple(np.array([node[i] for node in nodes]) for i in range(3))
        seg[coords] = seg_id

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(seg_to_vol, seg_id) for seg_id in range(ds_in.shape[0])]
        [t.result() for t in tasks]

    ds_out = f.require_dataset(out_key, shape=shape, chunks=chunks,
                               compression='gzip', dtype='uint64')
    ds_out.n_threads = n_threads
    ds_out[:] = seg


if __name__ == '__main__':
    # TODO which scale ?
    scale = 2
    # check_scale(scale)
    target = 'local'
    max_jobs = 48
    # skeletonize(scale, target, max_jobs)
    skeletons_to_volume(scale, max_jobs)
