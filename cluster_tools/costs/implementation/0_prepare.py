#! /usr/bin/python

import os
import argparse
import numpy as np
import z5py


def make_rf_jobs(tmp_folder, n_edges, chunk_size, n_jobs):
    n_chunks = n_edges // chunk_size + 1 if chunk_size % n_edges != 0 else n_edges // chunk_size
    assert n_jobs <= n_chunks, "%i, %i" % (n_jobs, n_chunks)

    # distribute chunks to jobs as equal as possible
    chunks_per_job = np.zeros(n_jobs, dtype='uint32')
    chunk_count = n_chunks
    job_id = 0
    while chunk_count > 0:
        chunks_per_job[job_id] += 1
        chunk_count -= 1
        job_id += 1
        job_id = job_id % n_jobs
    assert np.sum(chunks_per_job) == n_chunks

    chunk_index = 0
    for job_id in range(n_jobs):
        print(chunk_index, chunks_per_job[job_id])
        edge_begin = chunk_index * chunk_size
        edge_end = (chunk_index + chunks_per_job[job_id]) * chunk_size
        chunk_index += chunks_per_job[job_id]
        if job_id == n_jobs - 1:
            edge_end = n_edges
        # print(job_id, edge_begin, edge_end)
        np.save(os.path.join(tmp_folder, "1_input_%i.npy" % job_id),
                np.array([edge_begin, edge_end], dtype='uint32'))


def prepare(features_path, features_key,
            out_path, out_key,
            n_jobs, tmp_folder,
            random_forest_path=''):
    assert os.path.exists(features_path), features_path

    n_edges = z5py.File(features_path)[features_key].shape[0]
    edge_chunks = z5py.File(features_path)[features_key].chunks[0]

    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

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

    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('features_path', type=str)
    parser.add_argument('features_key', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('out_key', type=str)

    parser.add_argument('--n_jobs', type=int)
    parser.add_argument('--tmp_folder', type=str)
    parser.add_argument('--random_forest_path', type=str, default='')

    args = parser.parse_args()
    prepare(args.features_path, args.features_key,
            args.out_path, args.out_key, args.n_jobs,
            args.tmp_folder, args.random_forest_path)
