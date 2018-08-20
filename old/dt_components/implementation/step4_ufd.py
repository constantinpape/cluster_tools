#! /usr/bin/python

import argparse
import os
import numpy as np
import vigra
import nifty


def step4_ufd(cache_folder, n_jobs):

    assignments = []
    for job_id in range(n_jobs):
        res_path = os.path.join(cache_folder, '3_results_%i.npy' % job_id)
        assignment = np.load(res_path)
        if assignment.size > 0:
            assignments.append(assignment)
    assignments = np.concatenate(assignments, axis=0)
    n_labels = int(assignments.max()) + 1

    # stitch the segmentation (node level)
    ufd = nifty.ufd.ufd(n_labels)
    ufd.merge(assignments)
    node_labels = ufd.elementLabeling()
    vigra.analysis.relabelConsecutive(node_labels, keep_zeros=True,
                                      start_label=1, out=node_labels)

    out_path = os.path.join(cache_folder, 'node_assignments.npy')
    np.save(out_path, node_labels)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    step4_ufd(args.cache_folder, args.n_jobs)
