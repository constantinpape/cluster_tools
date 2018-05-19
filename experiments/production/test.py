import os
import json
import luigi
from production.components import Workflow


def write_config():
    with open('config.json', 'w') as f:
        json.dump({'boundary_threshold': 0.2,
                   'block_shape': [50, 512, 512],
                   'aff_slices': [[0, 12], [12, 13]],
                   'invert_channels': [True, False],
                   'chunks': [25, 256, 256]}, f)


def run_components(path, tmp_folder):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_folder, 'config.json')
    luigi.run(['--local-scheduler',
               '--path', path,
               '--aff-key', 'predictions/affs_glia',
               '--mask-key', 'masks/minfilter_mask',
               '--out-key', 'segmentation/components_test',
               '--max-jobs', '64',
               '--config-path', config_path,
               '--tmp-folder', tmp_folder,
               '--time-estimate', '10'], Workflow)

if __name__== '__main__':
    write_config()
    sample = 'A+'
    path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    tmp_folder = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/cremi_%s_production' % sample
    run_components(path, tmp_folder)
