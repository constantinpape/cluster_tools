import os
from scipy.ndimage.filters import minimum_filter
from concurrent import futures
import z5py
import nifty


def min_filter_mask(path, mask_key, out_key,
                    filter_shape,
                    chunks, blocks, n_threads=8):
    assert os.path.exists(path), path
    f = z5py.File(path, use_zarr_format=False)
    ds_mask = f[mask_key]
    halo = list(fshape // 2 + 1 for fshape in filter_shape)
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(ds_mask.shape),
                                    blockShape=list(blocks))

    if out_key not in f:
        ds = f.create_dataset(out_key,
                              shape=ds_mask.shape,
                              chunks=chunks,
                              dtype='uint8',
                              compression='gzip')
    else:
        ds = f[out_key]
        assert ds.shape == ds_mask.shape
        assert ds.chunks == chunks
    ds.attrs['minFilterKernel'] = filter_shape[::-1]

    def mask_block(block_id):
        print("Making min-filter mask for block", block_id, "/", blocking.numberOfBlocks)
        block = blocking.getBlockWithHalo(block_id, halo)
        outer_roi = tuple(slice(beg, end)
                          for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
        inner_roi = tuple(slice(beg, end)
                          for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
        local_roi = tuple(slice(beg, end)
                          for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
        mask = ds_mask[outer_roi]
        min_filter_mask = minimum_filter(mask, size=filter_shape)
        ds[inner_roi] = min_filter_mask[local_roi]

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(mask_block, block_id) for block_id in range(blocking.numberOfBlocks)]
        [t.result() for t in tasks]


if __name__ == '__main__':
    for sample in ('B+',):
        path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample
        mask_key = 'masks/original_mask'

        out_key = 'masks/minfilter_mask'
        chunks = (26, 256, 256)
        blocks = (52, 512, 512)

        net_in_shape = (88, 808, 808)
        net_out_shape = (60, 596, 596)
        filter_shape = tuple((netin - netout)
                             for netin, netout in zip(net_in_shape, net_out_shape))
        print(filter_shape)
        # filter_shape = tuple(fshape * 2 + 1 for fshape in filter_shape)

        min_filter_mask(path, mask_key, out_key,
                        filter_shape=filter_shape,
                        chunks=chunks,
                        blocks=blocks,
                        n_threads=30)
