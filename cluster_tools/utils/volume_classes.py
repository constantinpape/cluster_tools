from functools import partial
from math import floor, ceil

import numpy as np
import vigra
from scipy.ndimage import affine_transform

from .transformation_utils import compute_affine_matrix, transform_roi


# TODO use more fancy implementation (e.g. from z5)
def _normalize_index(index, shape):
    if isinstance(index, slice):
        index = (index,)
    else:
        assert isinstance(index, tuple)
        assert len(index) <= len(shape)
        assert all(isinstance(ind, slice) for ind in index)

    if len(index) < len(shape):
        n_missing = len(shape) - len(index)
        index = index + n_missing * (slice(None),)
    index = tuple(slice(0 if ind.start is None else ind.start,
                        sh if ind.stop is None else ind.stop)
                  for ind, sh in zip(index, shape))
    return index


# we need to support origin shifts,
# but it's probably better to do this with the affine matrix already
class TransformedVolume:
    """ Apply affine transformation to the volume.

    Arguments:
        volume: volume to which to apply the affine.
        output_shape: output shape, deduced from data by default (default: None)
        affine_matrix: matrix defining the affine transformation
        scale: scale factor
        rotation: rotation in degrees
        shear: shear in degrees
        translation: translation in pixel
        order: order of interpolation (supports 0 up to 5)
        fill_value: fill value for invalid regions (default: 0)
    """
    def __init__(self, volume, output_shape=None,
                 affine_matrix=None,
                 scale=None, rotation=None, shear=None, translation=None,
                 order=0, fill_value=0):

        # TODO support 2d + channels and 3d + channels
        assert volume.ndim in (2, 3), "Only 2d or 3d supported"
        self._volume = volume
        self._dtype = volume.dtype
        self._ndim = volume.ndim

        # scipy transformation options
        self.order = order
        self.fill_value = fill_value

        # validate the affine parameter
        have_matrix = affine_matrix is not None
        have_parameter = translation is not None or scale is not None or\
            rotation is not None or shear is not None

        if not (have_matrix != have_parameter):
            raise RuntimeError("Exactly one of affine_matrix or affine parameter needs to be passed")

        # get the affine matrix
        if have_matrix:
            self.matrix = affine_matrix
        else:
            assert shear is None, "Shear is not properly implemented yet"
            self.matrix = compute_affine_matrix(scale, rotation, shear, translation)
        assert self.matrix.shape == (self.ndim + 1, self.ndim + 1), "Invalid affine matrix"

        # TODO handle linalg inversion errors
        # get the inverse matrix
        self.inverse_matrix = np.linalg.inv(self.matrix)

        # comptue the extent and where the origin is mapped to in the target space
        extent, origin = self.compute_extent_and_origin()
        self.origin = origin

        # compute the shape after interpolation
        if output_shape is None:
            self._shape = extent
        else:
            assert isinstance(output_shape, tuple)
            assert len(output_shape) == self.ndim
            self._shape = output_shape

    @property
    def shape(self):
        return self._shape

    @property
    def volume(self):
        return self._volume

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return self._ndim

    def compute_extent_and_origin(self):
        roi_start, roi_stop = transform_roi([0] * self.ndim, self.volume.shape, self.matrix)
        extent = tuple(int(sto - sta) for sta, sto in zip(roi_start, roi_stop))
        return extent, roi_start

    def crop_to_input_space(self, roi_start, roi_stop):
        return ([max(rs, 0) for rs in roi_start],
                [min(rs, sh) for rs, sh in zip(roi_stop, self.volume.shape)])

    # TODO this seems to be correct for the full volume now, but not for cutouts yet
    def compute_offset(self, roi_start):
        return [-(sta + orig) for sta, orig in zip(roi_start, self.origin)]

    def __getitem__(self, index):
        # 1.) normalize the index to have a proper bounding box
        index = _normalize_index(index, self.shape)

        # 2.) transform the bounding box back to the input shape
        # (= coordinate system of self.volume) and make out shape
        roi_start, roi_stop = [ind.start for ind in index], [ind.stop for ind in index]
        tr_start, tr_stop = transform_roi(roi_start, roi_stop, self.inverse_matrix)
        out_shape = tuple(sto - sta for sta, sto in zip(roi_start, roi_stop))

        # 3.) crop the transformed bounding box to the valid region
        # and load the input data
        tr_start_cropped, tr_stop_cropped = self.crop_to_input_space(tr_start, tr_stop)
        transformed_index = tuple(slice(int(sta), int(sto))
                                  for sta, sto in zip(tr_start_cropped, tr_stop_cropped))
        input_ = self.volume[transformed_index]

        # TODO this seems to be correct for the full volume now, but not for cutouts yet
        # 4.) adapt the matrix for the local cutout
        tmp_mat = self.matrix.copy()
        offset = self.compute_offset(roi_start)
        tmp_mat[:self.ndim, self.ndim] = offset
        tmp_mat = np.linalg.inv(tmp_mat)

        # 5.) apply the affine transformation
        out = affine_transform(input_, tmp_mat, output_shape=out_shape,
                               order=self.order, mode='constant', cval=self.fill_value)

        return out

    def __setitem__(self, index, item):
        raise NotImplementedError("Setitem not implemented")


class InterpolatedVolume:
    """ Interpolate volume to a different shape.

    Arguments:
        volume [np.ndarray]: input volume
        output_shape [tuple]: target shape for interpolation
        spline_order [int]: order used for interpolation
    """
    def __init__(self, volume, output_shape, spline_order=0):
        assert len(output_shape) == volume.ndim == 3, "Only 3d supported"
        self.volume = volume
        self.shape = output_shape
        self.dtype = volume.dtype
        self.scale = [sh / float(fsh) for sh, fsh in zip(self.volume.shape, self.shape)]

        if np.dtype(self.dtype) == np.bool:
            self.min, self.max = 0, 1
        else:
            try:
                self.min = np.iinfo(np.dtype(self.dtype)).min
                self.max = np.iinfo(np.dtype(self.dtype)).max
            except ValueError:
                self.min = np.finfo(np.dtype(self.dtype)).min
                self.max = np.finfo(np.dtype(self.dtype)).max

        self.interpol_function = partial(vigra.sampling.resize, order=spline_order)

    def _interpolate(self, data, shape):
        # vigra can't deal with singleton dimensions, so we need to handle this seperately
        have_squeezed = False
        # check for singleton axes
        singletons = tuple(sh == 1 for sh in data.shape)
        if any(singletons):
            assert all(sh == 1 for is_single, sh in zip(singletons, shape) if is_single)
            inflate = tuple(slice(None) if sh > 1 else None for sh in data.shape)
            data = data.squeeze()
            shape = tuple(sh for is_single, sh in zip(singletons, shape) if not is_single)
            have_squeezed = True

        data = self.interpol_function(data.astype('float32'), shape=shape)
        np.clip(data, self.min, self.max, out=data)

        if have_squeezed:
            data = data[inflate]
        return data.astype(self.dtype)

    def __getitem__(self, index):
        index = _normalize_index(index, self.shape)
        # get the return shape and singletons
        ret_shape = tuple(ind.stop - ind.start for ind in index)
        singletons = tuple(sh == 1 for sh in ret_shape)

        # get the donwsampled index; respecting singletons
        starts = tuple(int(floor(ind.start * sc)) for ind, sc in zip(index, self.scale))
        stops = tuple(sta + 1 if is_single else int(ceil(ind.stop * sc))
                      for ind, sc, sta, is_single in zip(index, self.scale,
                                                         starts, singletons))
        index_ = tuple(slice(sta, sto) for sta, sto in zip(starts, stops))

        # check if we have a singleton in the return shape

        data_shape = tuple(idx.stop - idx.start for idx in index_)
        # remove singletons from data iff axis is not singleton in return data
        index_ = tuple(slice(idx.start, idx.stop) if sh > 1 or is_single else
                       slice(idx.start, idx.stop + 1)
                       for idx, sh, is_single in zip(index_, data_shape, singletons))
        data = self.volume[index_]

        # speed ups for empty blocks and masks
        dsum = data.sum()
        if dsum == 0:
            return np.zeros(ret_shape, dtype=self.dtype)
        elif dsum == data.size:
            return np.ones(ret_shape, dtype=self.dtype)
        return self._interpolate(data, ret_shape)

    def __setitem__(self, index, item):
        raise NotImplementedError("Setitem not implemented")
