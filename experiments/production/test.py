import os
import json
import luigi
from production import Workflow2DWS, write_default_config


def write_config():
    with open('config.json', 'w') as f:
        json.dump({'boundary_threshold': 0.15,
                   'block_shape': [50, 512, 512],
                   'block_shape2': [75, 768, 768],
                   'block_shift': [37, 384, 384],
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
                   'weight_merge_edges': False,
                   'weight_multicut_edges': False,
                   'affinity_offsets': [[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, -1]],
                   'use_lmc': False,
                   'rf_path': ['./rf_local_xy.pkl',
                               './rf_local_z.pkl'],  # './lifted_rf.pkl',
                   'lifted_nh': 2}, f)


def run_components(path, tmp_folder):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_folder, 'config.json')
    luigi.run(['--local-scheduler',
               '--path', path,
               '--aff-key', 'predictions/affs_glia',
               '--mask-key', 'masks/minfilter_mask',
               '--ws-key', 'segmentation/ws_test_2d',
               '--seg-key', 'segmentation/mc_seg_test_ws2d_rf',
               '--max-jobs', '64',
               '--config-path', config_path,
               '--tmp-folder-ws', tmp_folder,
               '--tmp-folder-seg', tmp_folder,
               '--time-estimate', '10',
               '--run-local'], Workflow2DWS)


if __name__ == '__main__':
    write_config()
    sample = 'A+'
    path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    tmp_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/cremi_%s_production' % sample
    run_components(path, tmp_folder)
