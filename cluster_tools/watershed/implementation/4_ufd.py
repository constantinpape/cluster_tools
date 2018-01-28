import os
import time
import argparse
import nifty
import numpy as np


def watershed_step4(tmp_folder, n_jobs):
    t0 = time.time()
    node_assignment = np.concatenate([np.load(os.path.join(tmp_folder, '3_output_assignments_%i.npy' % job_id))
                                     for job_id in range(n_jobs)], axis=0)
    max_id = np.load(os.path.join(tmp_folder, 'max_id.npy'))
    ufd = nifty.ufd.ufd(max_id + 1)
    ufd.merge(node_assignment)
    node_labeling = ufd.elementLabeling()
    np.save(os.path.join(tmp_folder, 'node_labeling.npy'), node_labeling)
    print("Success")
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_arguments("tmp_folder", type=str)
    parser.add_arguments("n_jobs", type=int)
    args = parser.add_arguments()
    watershed_step4(args.tmp_folder, args.n_jobs)
