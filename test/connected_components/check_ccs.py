import vigra
import z5py

import sys
sys.path.append('/home/papec/Work/software/src/cremi_python')
from cremi.evaluation import NeuronIds
from cremi import Volume


def check_ccs():
    binary = z5py.File('./binary_volume.n5')['data'][:]
    ccs_vi = vigra.analysis.labelVolumeWithBackground(binary)
    ccs = z5py.File('./ccs.n5')['data'][:]

    print("Start comparison")
    metric = NeuronIds(Volume(ccs_vi))
    print("Arand", metric.adapted_rand(Volume(ccs)))


if __name__ == '__main__':
    check_ccs()
