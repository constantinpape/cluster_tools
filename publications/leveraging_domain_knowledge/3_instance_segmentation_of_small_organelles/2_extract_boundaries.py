import z5py
from copy import deepcopy
import nifty.tools as nt
from concurrent import futures


def extract_boundaries(path, in_key, out_key):
    with z5py.File(path) as f:
        ds_in = f[in_key]
        shape = ds_in.shape[1:]
        chunks = ds_in.chunks[1:]
        blocks = deepcopy(chunks)
        chunks = tuple(ch // 4 for ch in chunks)
        ds_out = f.require_dataset(out_key, shape=shape, chunks=chunks, dtype='float32',
                                   compression='gzip')

        blocking = nt.blocking([0, 0, 0], list(shape), list(blocks))

        def copy_bound(block_id):
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            bb_in = (slice(1, 2),) + bb
            data = ds_in[bb_in].squeeze()
            ds_out[bb] = data

        print("Start extracting boundaries")
        with futures.ThreadPoolExecutor(8) as tp:
            tasks = [tp.submit(copy_bound, block_id)
                     for block_id in range(blocking.numberOfBlocks)]
            [t.result() for t in tasks]


if __name__ == '__main__':
    path = '/g/kreshuk/data/arendt/sponge/data.n5'
    extract_boundaries(path, 'volumes/predictions/semantic', 'volumes/predictions/boundaries')
