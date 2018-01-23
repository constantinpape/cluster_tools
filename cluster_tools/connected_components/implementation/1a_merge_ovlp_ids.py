#! /usr/bin/python

import os
import argparse
import numpy as np


def merge_ovlp_ids(tmp_folder, n_jobs):
    overlap_ids = []
    max_ids = []
    for job_id in range(n_jobs):
        overlap_ids.append(np.load(os.path.join(tmp_folder, '1_output_ovlps_%i.npy' % job_id)))
        max_ids.append(np.load(os.path.join(tmp_folder, '1_output_maxid_%i.npy' % job_id)))
    max_id = int(np.max(max_ids))
    overlap_ids = np.array(overlap_ids)

    chunk_size = len(overlap_ids) // n_jobs
    for i in range(0, len(overlap_ids), chunk_size):
        np.save(overlap_ids[i:i + chunk_size],
                os.path.join(tmp_folder, '2_input_%i.npy' % i))
    np.save(max_id, os.path.join(tmp_folder, "max_id.npy"))
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tmp_folder", str)
    parser.add_argument("n_jobs", int)
    args = parser.parse_args()
    merge_ovlp_ids(args.tmp_folder, args.n_jobs)
