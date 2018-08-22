import os
import json
import luigi
import z5py
from production import SegmentationWorkflow, write_default_config


def write_config(stitcher, roi=None):
    with open('config.json', 'w') as f:
        json.dump({'boundary_threshold': 0.15,
                   'block_shape': [50, 512, 512],
                   'block_shape2': [25, 256, 256],
                   'block_shift': [12, 128, 128],
                   'aff_slices': [[0, 12], [12, 13]],
                   'invert_channels': [True, False],
                   'chunks': [25, 256, 256],
                   'use_dt': False,
                   'resolution': (40, 4, 4),
                   'distance_threshold': 40,
                   'sigma': 2.,
                   'boundary_threshold2': 0.2,
                   'sigma_maxima': 2.6,
                   'size_filter': 25,
                   'n_threads': 16,
                   'merge_threshold': .8,
                   'weight_merge_edges': True,
                   'weight_multicut_edges': False,
                   'affinity_offsets': [[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, -1]],
                   'use_lmc': False,
                   'rf_path': ['./rf_local_xy.pkl',
                               './rf_local_z.pkl'],  # './lifted_rf.pkl',
                   # 'rf_path': None,
                   'lifted_nh': 2,
                   'roi': roi,
                   'stitch_task': stitcher}, f)


def run_components(path, tmp_folder, stitcher):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_folder, 'config.json')
    luigi.run(['--local-scheduler',
               '--path', path,
               '--aff-key', 'volumes/predictions/affinities',
               '--mask-key', 'volumes/labels/mask',
               '--ws-key', 'segmentation/ws_test',
               '--node-labeling-key', 'node_labels/labels_' + stitcher,
               '--seg-key', 'segmentation/seg_test_' + stitcher,
               '--max-jobs', '64',
               '--config-path', config_path,
               '--tmp-folder-ws', tmp_folder,
               '--tmp-folder-seg', tmp_folder,
               '--time-estimate', '10',
               '--run-local'], SegmentationWorkflow)


def get_central_roi(path, halo=[50, 512, 512]):
    shape = z5py.File(path)['raw'].shape
    beg = [sh // 2 - ha for sh, ha in zip(shape, halo)]
    end = [sh // 2 + ha for sh, ha in zip(shape, halo)]
    return [beg, end]


if __name__ == '__main__':
    sample = 'A+'
    # path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    path = '/nrs/saalfeld/papec/cremi2.0/training_data/V1_20180419/n5/sampleA.n5'
    # tmp_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/cremi_%s_production' % sample
    tmp_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/cremi_test'
    # roi = get_central_roi(path)
    stitcher = 'multicut'
    # stitcher = 'consensus_stitching'
    write_config(stitcher)
    run_components(path, tmp_folder, stitcher)
