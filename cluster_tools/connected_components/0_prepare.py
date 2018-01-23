import os
import argparse
import z5py
import json


def make_out_list(shape, block_shape, save_file):
    pass


def prepare(in_path, in_key,
            out_path, out_key,
            tmp_folder, out_block_shape,
            out_chunks):

    # TODO assert that out_block_shape is a multiple of out_chunks
    assert os.path.exists(in_path), in_path
    n5_in = z5py.File(in_path, use_zarr_format=False)
    ds = n5_in[in_key]
    shape = ds.shape

    n5_out = z5py.File(out_path, use_zarr_format=False)
    if out_key not in n5_out:
        ds_out = n5_out.create_dataset(out_key, dtype='uint64',
                                       shape=shape, chunks=out_chunks,
                                       compression='gzip')
    else:
        ds_out = n5_out[out_key]
        assert ds_out.shape == shape
        assert ds_out.chunks == out_chunks

    save_file = ''
    make_out_list(shape, out_block_shape, save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO read all arguments
    parser.add_argument("")
    prepare()
