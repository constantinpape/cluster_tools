import os
from concurrent import futures
from itertools import product

import numpy as np
import imageio
from z5py.shape_utils import normalize_slices


class KnossosDataset(object):
    block_size = 128

    @staticmethod
    def _chunks_dim(dim_root):
        files = os.listdir(dim_root)
        files = [f for f in files if os.path.isdir(os.path.join(dim_root, f))]
        return len(files)

    def get_shape_and_grid(self):
        cx = self._chunks_dim(self.path)
        y_root = os.path.join(self.path, 'x0000')
        cy = self._chunks_dim(y_root)
        z_root = os.path.join(y_root, 'y0000')
        cz = self._chunks_dim(z_root)

        grid = (cz, cy, cx)
        shape = tuple(sh * self.block_size for sh in grid)
        return shape, grid

    def __init__(self, path, file_prefix, load_png):
        self.path = path
        self.ext = 'png' if load_png else 'jpg'
        self.file_prefix = file_prefix

        self._ndim = 3
        self._chunks = self._ndim * (self.block_size,)
        self._shape, self._grid = self.get_shape_and_grid()
        self.n_threads = 1

    @property
    def dtype(self):
        return 'uint8'

    @property
    def ndim(self):
        return self._ndim

    @property
    def chunks(self):
        return self._chunks

    @property
    def shape(self):
        return self._shape

    def load_block(self, grid_id):
        # NOTE need to reverse grid id, because knossos folders are stored in x, y, z order
        block_path = ['%s%04i' % (dim, gid) for dim, gid in zip(('x', 'y', 'z'),
                                                                grid_id[::-1])]
        dim_str = '_'.join(block_path)
        fname = '%s_%s.%s' % (self.file_prefix, dim_str, self.ext)
        block_path.append(fname)
        path = os.path.join(self.path, *block_path)
        data = np.array(imageio.imread(path)).reshape(self._chunks)
        return data

    def _load_roi(self, roi):
        # snap roi to grid
        ranges = [range(rr.start // self.block_size,
                        rr.stop // self.block_size if
                        rr.stop % self.block_size == 0 else rr.stop // self.block_size + 1)
                  for rr in roi]
        grid_points = product(*ranges)

        # init data (dtype is hard-coded to uint8)
        roi_shape = tuple(rr.stop - rr.start for rr in roi)
        data = np.zeros(roi_shape, dtype='uint8')

        def load_tile(grid_id):

            grid_pos = [gid * self.block_size for gid in grid_id]
            tile_data = self.load_block(grid_id)
            # check how the tile-date fits into data
            left_offset = [rr.start - gp for rr, gp in zip(roi, grid_pos)]
            has_left_offset = [0 < lo < self.block_size for lo in left_offset]

            right_offset = [(gp + self.block_size) - rr.stop for rr, gp in zip(roi, grid_pos)]
            has_right_offset = [0 < ro < self.block_size for ro in right_offset]
            complete_overlap = not (any(has_left_offset) or any(has_right_offset))

            # 1.) complete overlap
            if complete_overlap:
                tile_bb = np.s_[:]
                out_bb = tuple(slice(gp - rr.start, gp + self.block_size - rr.start)
                               for rr, gp in zip(roi, grid_pos))
            # 2.) partial overlap
            else:
                tile_bb = []
                out_bb = []
                for ii in range(3):
                    if has_left_offset[ii]:
                        off = left_offset[ii]
                        tile_bb.append(slice(off, self.block_size))
                        out_bb.append(slice(0, self.block_size - off))
                    elif has_right_offset[ii]:
                        off = right_offset[ii]
                        dimlen = roi_shape[ii]
                        tile_bb.append(slice(0, off))
                        out_bb.append(slice(dimlen - off, dimlen))
                    else:
                        tile_bb.append(slice(None))
                        out_start = grid_pos[ii] - roi[ii].start
                        out_bb.append(slice(out_start, out_start + self.block_size))

                tile_bb = tuple(tile_bb)
                out_bb = tuple(out_bb)

            data[out_bb] = tile_data[tile_bb]

        if self.n_threads > 1:
            with futures.ThreadPoolExecutor(self.n_threads) as tp:
                tasks = [tp.submit(load_tile, sp) for sp in grid_points]
                [t.result() for t in tasks]
        else:
            [load_tile(sp) for sp in grid_points]
        #
        return data

    def __getitem__(self, index):
        roi = normalize_slices(index, self.shape)
        return self._load_roi(roi)


class KnossosFile(object):
    """ Wrapper for knossos file structure
    """
    def __init__(self, path, load_png=True):
        if not os.path.exists(os.path.join(path, 'mag1')):
            raise RuntimeError("Not a knossos file structure")
        self.path = path
        self.load_png = load_png
        self.file_name = os.path.split(self.path)[1]

    def __getitem__(self, key):
        ds_path = os.path.join(self.path, key)
        if not os.path.exists(ds_path):
            raise ValueError("Invalid key %s" % key)
        file_prefix = '%s_%s' % (self.file_name, key)
        return KnossosDataset(ds_path, file_prefix, self.load_png)
