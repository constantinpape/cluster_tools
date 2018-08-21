import sys
import luigi

import h5py
import z5py

from cremi_tools.viewer.volumina import view

try:
    from cluster_tools.watershed import WatershedLocal
except ImportError:
    sys.path.append('../..')
    from cluster_tools.watershed import WatershedLocal


def test_ws():
    input_path = '/home/cpape/Work/data/isbi2012/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5'
    input_key = output_key = 'data'
    output_path = './ws.n5'

    max_jobs = 8

    luigi.build([WatershedLocal(input_path=input_path, input_key=input_key,
                                output_path=output_path, output_key=output_key,
                                global_config_path='./test_config.json',
                                ws_config_path='./test_config.json',
                                tmp_folder='./tmp',
                                max_jobs=max_jobs)], local_scheduler=True)

    with h5py.File(input_path) as f:
        affs = f['data'][:3].transpose((1, 2, 3, 0))
    ws = z5py.File(output_path)['data'][:]
    view([affs, ws])


if __name__ == '__main__':
    test_ws()
