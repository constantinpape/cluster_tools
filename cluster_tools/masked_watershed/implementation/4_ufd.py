#! /usr/bin/python

import os
import time
import argparse
import nifty
import numpy as np


def masked_watershed_step4(tmp_folder, n_jobs):
    t0 = time.time()
    # load the node assignments and filter invalid assignments
    node_assignment = [np.load(os.path.join(tmp_folder, '3_output_assignments_%i.npy' % job_id))
                       for job_id in range(n_jobs)]
    node_assignment = np.concatenate([assignment for assignment in node_assignment if assignment.size > 0], axis=0)
    max_id = int(np.load(os.path.join(tmp_folder, 'max_id.npy')))

    ufd = nifty.ufd.ufd(max_id + 1)
    ufd.merge(node_assignment)
    node_labeling = ufd.elementLabeling()
    # make sure 0 gets mapped to 0
    if node_labeling[0] != 0:
        zero_label = node_labeling[0]
        labeled = np.where(node_labeling == zero_label)[0]
        assert len(labeled) == 1, "More than one node mapped to ignore label"
        node_labeling[node_labeling == 0] = zero_label
        node_labeling[0] = 0

    np.save(os.path.join(tmp_folder, 'node_labeling.npy'), node_labeling)
    print("Success")
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tmp_folder", type=str)
    parser.add_argument("n_jobs", type=int)
    args = parser.parse_args()
    masked_watershed_step4(args.tmp_folder, args.n_jobs)
