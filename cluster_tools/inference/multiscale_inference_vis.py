import dask
import toolz as tz
import numpy as np
import nifty.tools as nt
from elf.io import open_file

from .multiscale_inference import _show_inputs, _load_inputs


def view_multiscale_inputs(input_path, input_group, input_scales, scale_factors, halos, block_shape,
                           n_inputs=None, randomize_inputs=True, block_list=None, n_threads=8):
    assert len(input_scales) == len(scale_factors) == len(halos), "%i, %i, %i" % (len(input_scales),
                                                                                  len(scale_factors),
                                                                                  len(halos))
    assert (n_inputs is None) != (block_list is None)

    f = open_file(input_path, 'r')
    g = f[input_group]
    datasets = [g[in_scale] for in_scale in input_scales]

    shape = datasets[0].shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    if n_inputs is not None:
        if randomize_inputs:
            n_blocks = blocking.numberOfBlocks
            block_list = np.random.choice(n_blocks, n_inputs, replace=False)
        else:
            block_list = range(n_inputs)

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
