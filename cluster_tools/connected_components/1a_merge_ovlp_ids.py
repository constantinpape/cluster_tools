import json
import os
import argparse
import numpy as np


def merge_ovlp_ids(tmp_folder, n_jobs):
    overlap_ids = []
    max_ids = []
    for job_id in range(n_jobs):
        with open(os.path.join(tmp_folder, '1_output_%i.json' % job_id), 'r') as f:
            inputs = json.load(f)
            max_ids.append(inputs['max_id'])
            overlap_ids.extend(inputs["overlap_ids"])
    max_id = int(np.max(max_ids))

    chunk_size = len(overlap_ids) // n_jobs
    for i in range(0, len(overlap_ids), chunk_size):
        with open(os.path.join(tmp_folder, '2_input_%i.json' % i), 'w') as f:
            json.dump({'job_id': i, 'overlap_ids': overlap_ids[i:i + chunk_size]}, f)
    with open(os.path.join(tmp_folder, "max_id.json"), 'w') as f:
        json.dump({'max_id': max_id})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tmp_folder", str)
    parser.add_argument("n_jobs", int)
    args = parser.parse_args()
    merge_ovlp_ids(args.tmp_folder, args.n_jobs)
