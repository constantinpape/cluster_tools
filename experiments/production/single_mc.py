import os
import luigi
import z5py
from production import Workflow2DWS, write_default_config
from cremi_tools.viewer.volumina import view


def run_mc(path, tmp_folder, roi):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_folder, 'config.json')
    write_default_config(config_path, roi=roi)
    luigi.run(['--local-scheduler',
               '--path', path,
               '--aff-key', 'predictions/affs_glia',
               '--mask-key', 'masks/minfilter_mask',
               '--ws-key', 'segmentation/ws_test_2d',
               '--seg-key', 'segmentation/mc_debug',
               '--max-jobs', '64',
               '--config-path', config_path,
               '--tmp-folder-ws', tmp_folder,
               '--tmp-folder-seg', tmp_folder,
               '--time-estimate', '10',
               '--run-local'], Workflow2DWS)


def get_central_roi(path, halo=[50, 512, 512]):
    shape = z5py.File(path)['raw'].shape
    beg = [sh // 2 - ha for sh, ha in zip(shape, halo)]
    end = [sh // 2 + ha for sh, ha in zip(shape, halo)]
    return [beg, end]


if __name__ == '__main__':
    sample = 'A+'
    path = '/home/papec/mnt/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
    tmp_folder = '/home/papec/mnt/papec/Work/neurodata_hdd/cache/cremi_%s_production' % sample
    roi = get_central_roi(path)
    print(roi)
    run_mc(path, tmp_folder, roi)
