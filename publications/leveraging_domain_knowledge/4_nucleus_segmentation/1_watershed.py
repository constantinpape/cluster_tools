import numpy as np
import vigra
import z5py

from cluster_tools.utils.volume_utils import normalize, watershed
from nifty.filters import nonMaximumDistanceSuppression
from cremi_tools.viewer.volumina import view


def _points_to_vol(points, shape):
    vol = np.zeros(shape, dtype='uint32')
    coords = tuple(points[:, i] for i in range(points.shape[1]))
    vol[coords] = 1
    return vigra.analysis.labelMultiArrayWithBackground(vol)


def dt_watershed(input_, threshold=.25, sigma=2., alpha=.9, min_seg_size=100, suppression=True):
    # make distance transform
    threshd = (input_ > threshold).astype('uint32')
    dt = vigra.filters.distanceTransform(threshd)
    dt = vigra.filters.gaussianSmoothing(dt, sigma)

    # make seeds
    seeds = vigra.analysis.localMaxima3D(dt, allowPlateaus=True, allowAtBorder=True,
                                         marker=np.nan)
    seeds = np.isnan(seeds).astype('uint32')
    seeds = vigra.analysis.labelVolumeWithBackground(seeds)
    # non-max suppression
    if suppression:
        seeds = np.array(np.where(seeds)).transpose()
        seeds = nonMaximumDistanceSuppression(dt, seeds)
        seeds = _points_to_vol(seeds, dt.shape)

    # make hmap
    hmap = alpha * input_ + (1. - alpha) * (1. - normalize(dt))

    ws, max_id = watershed(hmap, seeds, min_seg_size)
    return ws, max_id


def run_ws(timepoint, bb, save=False):
    path = '/g/kreshuk/pape/Work/data/lifted_priors/plant_data/t%03i.n5' % timepoint
    f = z5py.File(path)
    ds = f['volumes/predictions/boundaries']
    ds.n_threads = 16
    bd = ds[bb]

    print("run watershed for timepoint", timepoint, "...")
    ws, max_id = dt_watershed(bd)
    ws = ws.astype('uint64')

    chunks = (25, 256, 256)
    if save:
        ds = f.create_dataset('volumes/segmentation/watershed', shape=ws.shape,
                              dtype=ws.dtype, compression='gzip', chunks=chunks)
        ds.n_threads = 16
        ds[:] = ws
        ds.attrs['maxId'] = max_id
    else:
        ds = f['volumes/raw']
        raw = ds[bb]
        view([raw, bd, ws])


if __name__ == '__main__':
    bb = np.s_[:]
    # run_ws(45, bb, save=True)
    run_ws(49, bb, save=True)
