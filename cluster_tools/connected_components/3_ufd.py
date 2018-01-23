import json
import os
import numpy as np
import nifty


def cc_ufd_step3(tmp_folder, n_jobs):
    with open(os.path.join(tmp_folder, 'max_id.json'), 'r') as f:
        max_id = json.load(f)['max_id']

    node_assignment = []
    for job_id in range(n_jobs):
        with open() as f:
            node_assignment.append(json.load(f))
    node_assignment = np.array(node_assignment)

    ufd = nifty.ufd.ufd(max_id + 1)
    ufd.merge(node_assignment)
    # TODO do we need extra treatment for zeros ?
    node_labeling = ufd.elementLabeling()
# TODO
