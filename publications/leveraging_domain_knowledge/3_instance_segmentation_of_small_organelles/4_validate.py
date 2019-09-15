import os
import json
import numpy as np
import h5py
import z5py
import vigra

from cremi_tools.viewer.volumina import view
from cremi_tools.metrics import voi, adapted_rand


CENTRAL = {1: [32334, 40241, 23746],
           2: [48063, 50810, 23456],
           3: [28603, 44389, 39584]}


def get_bb(block_id):
    central = CENTRAL[block_id][::-1]
    halo = [32, 384, 384]
    resolution = [15, 15, 15]
    bb = tuple(slice(ce // re - ha, ce // re + ha)
               for ce, re, ha in zip(central, resolution, halo))
    return bb


def get_scores(seg, gt):
    rand = adapted_rand(seg, gt)
    vis, vim = voi(seg, gt)
    return {'rand': rand, 'vi-split': vis, 'vi-merge': vim}


def evaluate_seg(seg, gt, semantic):
    # evaluate overall segmentation
    scores_all = get_scores(seg, gt)
    res = {'all': scores_all}

    # evaluate semantic scores
    sem_ids = {1: 'cells', 2: 'flagella', 3: 'microvilli'}
    for sid, name in sem_ids.items():
        seg_sem, gt_sem = seg.copy(), gt.copy()
        sem_mask = semantic != sid
        seg_sem[sem_mask] = 0
        gt_sem[sem_mask] = 0
        res[name] = get_scores(seg_sem, gt_sem)

    return res


def validate_block(block_id):
    gt_path = '/g/kreshuk/data/arendt/sponge/nn_train_data/train_data_0%i.h5' % block_id
    with h5py.File(gt_path) as f:
        gt = f['volumes/labels/instances'][:]
        sem = f['volumes/labels/semantic'][:]
    mask = np.where(gt != -1)
    bb = tuple(slice(int(ma.min()), int(ma.max() + 1))
               for ma in mask)

    gt = vigra.analysis.labelVolumeWithBackground(gt[bb].astype('uint32'))
    sem = sem[bb]

    global_bb = get_bb(block_id)
    path = '/g/kreshuk/data/arendt/sponge/data.n5'
    f = z5py.File(path)
    keys = ['volumes/segmentation/for_eval/multicut',
            'volumes/segmentation/for_eval/lifted_multicut',
            'volumes/segmentation/for_eval/lifted_multicut_all']

    res = {}
    for key in keys:
        ds = f[key]
        ds.n_threads = 8
        seg = vigra.analysis.labelVolume(ds[global_bb].astype('uint32'))
        assert seg.shape == gt.shape
        eval_res = evaluate_seg(seg, gt, sem)
        res[os.path.split(key)[1]] = eval_res
    return res


def validate_all():
    res = {}
    for block_id in range(1, 4):
        print("Run validation for block", block_id)
        res[block_id] = validate_block(block_id)
    with open('./eval_results.json', 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)


def view_block(block_id):
    gt_path = '/g/kreshuk/data/arendt/sponge/nn_train_data/train_data_0%i.h5' % block_id
    with h5py.File(gt_path, 'r') as f:
        raw = f['volumes/raw'][:]
        gt = f['volumes/labels/instances'][:]
        sem = f['volumes/labels/semantic'][:]
    mask = np.where(gt != -1)
    bb = tuple(slice(int(ma.min()), int(ma.max() + 1))
               for ma in mask)

    raw = raw[bb]
    gt = gt[bb]
    sem = sem[bb]
    assert raw.shape == gt.shape == sem.shape
    print(raw.shape, gt.shape)

    global_bb = get_bb(block_id)
    print(global_bb)

    path = '/g/kreshuk/data/arendt/sponge/data.n5'
    f = z5py.File(path)
    ds_mc = f['volumes/segmentation/for_eval/multicut']
    seg_mc = vigra.analysis.labelVolume(ds_mc[global_bb].astype('uint32'))
    assert seg_mc.shape == gt.shape

    ds_lmc = f['volumes/segmentation/for_eval/lifted_multicut']
    seg_lmc = vigra.analysis.labelVolume(ds_lmc[global_bb].astype('uint32'))
    assert seg_lmc.shape == gt.shape

    view([raw, gt, sem, seg_mc, seg_lmc],
         ['raw', 'gt', 'sem', 'seg-mc', 'seg_lmc'])


if __name__ == '__main__':
    # view_block(3)
    # res = validate_block(1)
    # print(res)
    validate_all()
