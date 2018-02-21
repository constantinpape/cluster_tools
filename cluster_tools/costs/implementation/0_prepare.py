#! /usr/bin/python

import os
import argparse
import numpy as np
import z5py


def make_rf_jobs(tmp_folder, n_edges, chunk_size, n_jobs):
    n_chunks = n_edges // chunk_size + 1 if chunk_size % n_edges != 0 else n_edges // chunk_size
    assert n_jobs >= n_chunks, "%i, %i" % (n_jobs, n_chunks)

    chunks_per_job = n_chunks // n_jobs + 1 if n_chunks % n_jobs else n_jobs // n_chunks

    for job_id in range(n_jobs):
        edge_begin = job_id * chunks_per_job * chunk_size
        edge_end = min((job_id + 1) * chunks_per_job * chunk_size, n_edges)
        # if job_id == n_jobs - 1:
        #     edge_end = n_edges
        np.save(os.path.join(tmp_folder, "1_input_%i.npy" % job_id),
                np.array([edge_begin, edge_end], dtype='uint32'))


def prepare(features_path, features_key,
            out_path, out_key,
            n_jobs, tmp_folder,
            random_forest_path=''):
    assert os.path.exists(features_path), features_path

    n_edges = z5py.File(features_path)[features_key].shape[0]
    edge_chunks = z5py.File(features_path)[features_key].chunks[0]

    # if we use a random forest, we need to prepare the jobs for it
    # otherwise, we don't need to schedule any extra jobs
    if random_forest_path != '':
        assert os.path.exists(random_forest_path), random_forest_path
        make_rf_jobs(tmp_folder, n_edges, edge_chunks, n_jobs)

    # make the output costs dataset
    f = z5py.File(out_path, use_zarr_format=False)
    if out_key not in f:
        f.create_dataset(out_key, dtype='float32',
                         compression='gzip', shape=(n_edges,),
                         chunks=(edge_chunks,))
    else:
        ds = f[out_key]
        assert ds.shape == (n_edges,)
        assert ds.chunks == (edge_chunks,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('features_path', type=str)
    parser.add_argument('features_key', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('out_key', type=str)

    parser.add_argument('--n_jobs', type=str)
    parser.add_argument('--tmp_folder', type=str)
    parser.add_argument('--random_forest_path', type=str, default='')

    args = parser.parse_args()
    prepare(args.features_path, args.features_key,
            args.out_path, args.out_key, args.n_jobs,
            args.tmp_folder, args.random_forest_path)
