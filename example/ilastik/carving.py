import os
import luigi
import json

import numpy as np
import imageio
import vigra
import z5py
from cluster_tools.ilastik import IlastikCarvingWorkflow


def make_input():
    # path = '/home/pape/Work/data/ilastik/helmstaedter/images/Helmstaedter_et_al_SUPPLinformation3c.tif'
    path = '/g/kreshuk/pape/Work/data/ilastik_test_data/helmstaedter/images/Helmstaedter_et_al_SUPPLinformation3c.tif'
    hmap = np.asarray(imageio.volread(path)).astype('float32')
    hmap = hmap / hmap.max()
    hmap = 1. - hmap
    hmap = vigra.filters.gaussianSmoothing(hmap, 2.)

    ws, max_id = vigra.analysis.watershedsNew(hmap)
    ws = ws.astype('uint64')

    out_path = '/g/kreshuk/pape/Work/data/ilastik_test_data/helmstaedter/data.n5'
    with z5py.File(out_path) as f:
        f.create_dataset('volumes/hmap', data=hmap, compression='gzip', chunks=(64, 64, 64))
        ds = f.create_dataset('volumes/ws', data=ws, compression='gzip', chunks=(64, 64, 64))
        ds.attrs['maxId'] = max_id


def carving_wf():
    tmp_folder = 'tmp_carv'
    config_folder = 'conf_carv'
    os.makedirs(config_folder, exist_ok=True)

    configs = IlastikCarvingWorkflow.get_config()
    global_conf = configs['global']

    # shebang = '#! /home/pape/Work/software/conda/miniconda3/envs/main/bin/python'
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    block_shape = [64, 64, 64]
    global_conf.update({'shebang': shebang, 'block_shape': block_shape})
    with open('./conf_carv/global.config', 'w') as f:
        json.dump(global_conf, f)

    # path = '/home/pape/Work/data/ilastik/helmstaedter/data.n5'
    path = '/g/kreshuk/pape/Work/data/ilastik_test_data/helmstaedter/data.n5'

    task = IlastikCarvingWorkflow
    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=4, target='local',
             input_path=path, input_key='volumes/hmap',
             watershed_path=path, watershed_key='volumes/ws',
             output_path='./carv_test.ilp', copy_inputs=False)
    luigi.build([t], local_scheduler=True)


if __name__ == '__main__':
    # make_input()
    carving_wf()
