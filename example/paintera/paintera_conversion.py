import os
import json
from concurrent import futures

import numpy as np
import z5py
import luigi
import nifty.tools as nt
import nifty.distributed as ndist
from cluster_tools.paintera import ConversionWorkflow


def to_paintera_format():
    target = 'local'
    max_jobs = 8

    path = './data/data.n5'
    with z5py.File(path) as f:
        ds = f['labels']
        offset = ds.attrs['offset']
        resolution = ds.attrs['resolution']

    config_dir = './configs'
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    configs = ConversionWorkflow.get_config()

    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python"
    global_config = configs['global']
    global_config.update({'shebang': shebang})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_config, f)

    sampling_config = {'library': 'vigra', 'library_kwargs': {'order': 0}}

    ds_config = configs['downscaling']
    ds_config.update({**sampling_config})
    with open(os.path.join(config_dir, 'downscaling.config'), 'w') as f:
        json.dump(ds_config, f)

    t = ConversionWorkflow(path=path,
                           raw_key='volumes/raw',
                           label_in_key='labels',
                           label_out_key='volumes/labels/neuron_ids',
                           assignment_key='',
                           label_scale=0,
                           offset=offset,
                           resolution=resolution,
                           tmp_folder='./tmp',
                           max_jobs=max_jobs,
                           config_dir=config_dir,
                           target=target)
    luigi.build([t], local_scheduler=True)


def check_block_uniques(scale):
    print("Checking block uniques for scale", scale)
    path = './data/data.n5'
    f = z5py.File(path)
    ds_seg = f['volumes/labels/neuron_ids/data/s%i' % scale]
    ds_uns = f['volumes/labels/neuron_ids/unique-labels/s%i' % scale]

    print("Checking shapes and chunk shapes")
    assert ds_seg.shape == ds_uns.shape
    assert ds_seg.chunks == ds_uns.chunks
    chunks = list(ds_seg.chunks)

    blocking = nt.blocking([0, 0, 0], list(ds_seg.shape), chunks)

    def check_chunk(chunk_id):
        chunk = blocking.getBlock(chunk_id)
        chunk_coord = tuple(beg // ch for beg, ch in zip(chunk.begin, chunks))
        seg = ds_seg.read_chunk(chunk_coord)
        uns = ds_uns.read_chunk(chunk_coord)
        assert seg is not None
        assert uns is not None
        seg_uns = np.unique(seg)
        return np.allclose(uns, seg_uns)

    print("Checking correctness for %i chunks" % blocking.numberOfBlocks)
    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(check_chunk, chunk_id)
                 for chunk_id in range(blocking.numberOfBlocks)]
        results = [t.result() for t in tasks]

    assert len(results) == blocking.numberOfBlocks

    if not all(results):
        n_failed = len(results) - sum(results)
        print(n_failed, "/", len(results), "chunks falied")
        assert False

    print("All checks passed")


def check_all_uniques():
    for scale in range(3):
        check_block_uniques(scale)


def check_id(label_id, seg, rois):
    for roi in rois:
        bb = np.s_[roi[2]:roi[5] + 1,
                   roi[1]:roi[4] + 1,
                   roi[0]:roi[3] + 1]
        if label_id not in seg[bb]:
            return False
    return True


def check_block_mapping(scale):
    print("Checking block mapping for scale", scale)
    path = './data/data.n5'
    f = z5py.File(path)
    ds_seg = f['volumes/labels/neuron_ids/data/s%i' % scale]
    ds_map = f['volumes/labels/neuron_ids/label-to-block-mapping/s%i' % scale]

    ds_seg.n_threads = 8
    seg = ds_seg[:]

    chunk_id = (0,)
    while True:
        print("Checking mapping chunk", chunk_id)
        mapping = ndist.readBlockMapping(ds_map.path, chunk_id)

        if not mapping:
            print("Breaking at chunk", chunk_id)
            break
        print("Chunk has %i ids" % len(mapping))

        with futures.ThreadPoolExecutor(8) as tp:
            tasks = [tp.submit(check_id, label_id, seg, rois)
                     for label_id, rois in mapping.items()]
            results = [t.result() for t in tasks]
        assert len(results) == len(mapping)

        if not all(results):
            n_failed = len(results) - sum(results)
            print(n_failed, "/", len(results), "ids in chunk falied")
            assert False
        chunk_id = (chunk_id[0] + 1,)

    print("All checks passed")


def check_all_mappings():
    for scale in range(3):
        check_block_mapping(scale)


if __name__ == '__main__':
    # check_all_mappings()
    # check_all_uniques()
    to_paintera_format()
