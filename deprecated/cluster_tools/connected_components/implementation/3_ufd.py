#! /usr/bin/python

import os
import argparse
import numpy as np
import nifty
import vigra


def cc_ufd_step3(tmp_folder, n_jobs):

    max_ids = []
    for job_id in range(n_jobs):
        max_ids.append(np.load(os.path.join(tmp_folder, '1_output_maxid_%i.npy' % job_id)))
    max_id = int(np.max(max_ids))

    node_assignment = []
    for job_id in range(n_jobs):
        node_assignment.append(np.load(os.path.join(tmp_folder,
                                                    '2_output_%i.npy' % job_id)).squeeze())
    node_assignment = np.concatenate(node_assignment, axis=0)

    ufd = nifty.ufd.ufd(max_id + 1)
    ufd.merge(node_assignment)
    node_labeling = ufd.elementLabeling()
    vigra.analysis.relabelConsecutive(node_labeling, out=node_labeling)

    # make sure that 0 is mapped to 0
    if not node_labeling[0] == 0:
        if 0 in node_labeling:
            zero_labeled = node_labeling == 0
            node_labeling[zero_labeled] = node_labeling[0]
            node_labeling[0] = 0

    np.save(os.path.join(tmp_folder, 'node_labeling.npy'), node_labeling)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tmp_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    cc_ufd_step3(args.tmp_folder, args.n_jobs)
