import os
import argparse
import json
import vigra
import numpy as np


def process_overlap(ovlp_ids, tmp_folder):
    id_a, id_b = ovlp_ids
    ovlp_a = vigra.readHDF5(os.path.join(tmp_folder, 'block_%i_%i.h5' % (id_a, id_b)), 'data')
    ovlp_b = vigra.readHDF5(os.path.join(tmp_folder, 'block_%i_%i.h5' % (id_b, id_a)), 'data')

    # match the non-zero ids
    labeled = ovlp_a != 0
    ids_a, ids_b = ovlp_a[labeled], ovlp_b[labeled]
    node_assignment = np.concatenate([ids_a[None], ids_b[None]], axis=0).transpose()
    if node_assignment.size:
        node_assignment = np.unique(node_assignment, axis=0)
        return node_assignment


def cc_ufd_step2(tmp_folder, ovlp_file):
    with open(ovlp_file, 'r') as f:
        inputs = json.load(f)
        job_id = inputs['job_id']
        overlap_ids = inputs['overlap_ids']

    node_assignments = [process_overlap(ovlp_ids) for ovlp_ids in overlap_ids]
    node_assignments = [na.tolist() for na in node_assignments if na is not None]
    with open(os.path.join(tmp_folder, '2_output_%i.json' % job_id), 'w') as f:
        json.dump(node_assignments, f)
    print("Success job %i" % job_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tmp_folder', str)
    parser.add_argument('ovlp_file', str)
    args = parser.parse_args()
    cc_ufd_step2(args.tmp_folder, args.ovlp_file)
