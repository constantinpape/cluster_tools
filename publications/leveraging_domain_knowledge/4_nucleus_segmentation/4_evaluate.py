import os
import numpy as np
import z5py
import h5py
import vigra

from cremi_tools.viewer.volumina import view
from elf.evaluation import cremi_score

BBS = {45: np.s_[100:300, 0:450, 500:1500],
       49: np.s_[100:300, 0:450, 500:1500]}


def load_seg_and_gt(timepoint, seg_key, load_raw=False):
    root_path = '/g/kreshuk/wolny/Datasets/Constantin_lifted_mc_test/corrected_gt_results'
    gt_path = os.path.join(root_path,
                           't%05i_s00_uint8.h5' % timepoint)

    bb = BBS[timepoint]
    with h5py.File(gt_path, 'r') as f:
        ds = f['gt_primordia']
        gt = ds[:]
        shape = gt.shape

    path = '/g/kreshuk/pape/Work/data/lifted_priors/plant_data/t%03i.n5' % timepoint
    with z5py.File(path, 'r') as f:
        ds = f[seg_key]
        ds.n_threads = 8
        seg = ds[bb]
        seg = vigra.analysis.labelVolumeWithBackground(seg.astype('uint32'))
    assert shape == seg.shape
    if load_raw:
        with z5py.File(path, 'r') as f:
            ds = f['volumes/raw']
            ds.n_threads = 8
            raw = ds[bb]
        return seg, gt, raw
    else:
        return seg, gt


def evaluate_seg(timepoint, seg_key):
    seg, gt = load_seg_and_gt(timepoint, seg_key)
    scores = cremi_score(seg, gt, ignore_gt=[0])
    print("vi-split  vi-merge  adapted-rand ")
    print(scores[:3])


def view_gt_crop(timepoint):
    mc_key = 'volumes/segmentation/multicut'
    lmc_key = 'volumes/segmentation/lifted_multicut_repulsive'
    seg_mc, gt, raw = load_seg_and_gt(timepoint, mc_key, True)

    bb = BBS[timepoint]
    path = '/g/kreshuk/pape/Work/data/lifted_priors/plant_data/t%03i.n5' % timepoint
    f = z5py.File(path)
    ds = f[lmc_key]
    ds.n_threads = 8
    seg_lmc = ds[bb]
    seg_lmc = vigra.analysis.labelVolumeWithBackground(seg_lmc.astype('uint32'))

    ds = f['volumes/segmentation/nuclei']
    ds.n_threads = 8
    seg_nuc = ds[bb]

    view([raw, seg_mc, seg_lmc, gt, seg_nuc],
         ['raw', 'seg-mc', 'seg-lmc', 'gt', 'nuclei'])


if __name__ == '__main__':
    tp = 45
    mc_key = 'volumes/segmentation/multicut'
    lmc_key = 'volumes/segmentation/lifted_multicut_repulsive'
    print("MC:")
    evaluate_seg(tp, mc_key)
    print("LMC:")
    evaluate_seg(tp, lmc_key)
    # view_gt_crop(tp)
