import dask
import toolz as tz
import numpy as np
import nifty.tools as nt
from elf.io import open_file

import cluster_tools.utils.volume_utils as vu
from cluster_tools.inference.multiscale_inference import _show_inputs, _load_inputs


def view_multiscale_inputs(input_path, input_group, input_scales,
                           scale_factors, halos, block_shape,
                           n_inputs=None, randomize_inputs=True,
                           block_list=None, n_threads=8,
                           roi_begin=None, roi_end=None):
    assert len(input_scales) == len(scale_factors) == len(halos), "%i, %i, %i" % (len(input_scales),
                                                                                  len(scale_factors),
                                                                                  len(halos))
    assert (n_inputs is None) != (block_list is None)

    f = open_file(input_path, 'r')
    g = f[input_group]
    datasets = [g[in_scale] for in_scale in input_scales]

    shape = datasets[0].shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    if (roi_begin is None) and (roi_end is None):
        all_blocks = np.arange(blocking.numberOfBlocks)
    else:
        all_blocks = np.array(vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end))

    if block_list is None:
        if randomize_inputs:
            np.random.shuffle(all_blocks)
        block_list = all_blocks[:n_inputs]
    else:
        assert len(np.setdiff1d(block_list, all_blocks)) == 0

    @dask.delayed
    def load_input(block_id):
        block = blocking.getBlock(block_id)
        inputs = _load_inputs(datasets, block.begin,
                              block_shape, halos, scale_factors)
        return block_id, inputs

    results = []
    for block_id in block_list:
        res = tz.pipe(block_id, load_input)
        results.append(res)
    results = dask.compute(*results, scheduler='threads', num_workers=n_threads)
    for block_id, inputs in results:
        print("Show inputs for block", block_id)
        _show_inputs(inputs, scale_factors)
