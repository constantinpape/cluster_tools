#! /usr/bin/python

import os
import argparse
import vigra
import numpy as np


def process_overlap(ovlp_ids, tmp_folder):
    id_a, id_b = ovlp_ids
    ovlp_a = vigra.readHDF5(os.path.join(tmp_folder, 'block_ovlp_%i_%i.h5' % (id_a, id_b)), 'data')
    ovlp_b = vigra.readHDF5(os.path.join(tmp_folder, 'block_ovlp_%i_%i.h5' % (id_b, id_a)), 'data')

    # match the non-zero ids
    labeled = ovlp_a != 0
    ids_a, ids_b = ovlp_a[labeled], ovlp_b[labeled]
    node_assignment = np.concatenate([ids_a[None], ids_b[None]], axis=0).transpose()
    if node_assignment.size:
        node_assignment = np.unique(node_assignment, axis=0)
        # filter zeros
        valid_assignments = (node_assignment != 0).all(axis=1)
        return node_assignment[valid_assignments]


def cc_ufd_step2(tmp_folder, ovlp_file):
    overlap_ids = np.load(ovlp_file).squeeze()
    job_id = int(os.path.split(ovlp_file)[1].split('_')[3][:-4])

    node_assignments = [process_overlap(ovlp_ids, tmp_folder) for ovlp_ids in overlap_ids]
    node_assignments = np.concatenate([na for na in node_assignments if na is not None],
                                      axis=0)
    np.save(os.path.join(tmp_folder, '2_output_%i.npy' % job_id), node_assignments)
    print("Success job %i" % job_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tmp_folder', type=str)
    parser.add_argument('ovlp_file', type=str)
    args = parser.parse_args()
    cc_ufd_step2(args.tmp_folder, args.ovlp_file)
