#! /usr/bin/python

import os
import argparse
import numpy as np
import nifty


def cc_ufd_step3(tmp_folder, n_jobs):
    max_id = np.load('max_id.npy')

    node_assignment = []
    for job_id in range(n_jobs):
        node_assignment.append(np.load(os.path.join('tmp_folder',
                                                    '2_output_%i.npy' % job_id)))
    node_assignment = np.array(node_assignment)

    ufd = nifty.ufd.ufd(max_id + 1)
    ufd.merge(node_assignment)
    # TODO do we need extra treatment for zeros ?
    node_labeling = ufd.elementLabeling()
    # TODO relabele node-labeling ?

    np.save(os.path.join(tmp_folder, 'node_labeling.npy'), node_labeling)

    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tmp_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    cc_ufd_step3(args.tmp_folder, args.n_jobs)
