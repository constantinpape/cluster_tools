#! /usr/bin/python

import time
import argparse
import subprocess
from wait_and_check import wait_and_check_multiple_jobs, wait_and_check_single_job


# TODO retrial for failed jobs
def master_job(n_jobs, prefixes):

    t_tot = time.time()
    # submit jobs 1
    for prefix in prefixes:
        subprocess.call(['./jobs_step1_%s.sh' % prefix])
        # wait for jobs 1, only if we have a random forest
        failed_jobs = wait_and_check_multiple_jobs('consensus_stitching_step1_%s' % prefix, n_jobs)
        if failed_jobs:
            print("Step 1 %s failed for following jobs:" % prefix)
            print(failed_jobs)
            return

    # submit jobs 2
    subprocess.call(['./jobs_step2.sh'])
    # wait for jobs 2
    failed_jobs = wait_and_check_single_job('costs_step2')
    if failed_jobs:
        print("Step 2 failed")
        return

    subprocess.call(['./jobs_step3.sh'])
    # wait for jobs 1, only if we have a random forest
    failed_jobs = wait_and_check_multiple_jobs('consensus_stitching_step3', n_jobs)
    if failed_jobs:
        print("Step 3 failed for following jobs:")
        print(failed_jobs)
        return

    t_tot = time.time() - t_tot
    print("All jobs finished successfully in %f s" % t_tot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('prefixes', type=str, nargs='+')
    args = parser.parse_args()
    master_job(args.n_jobs, tuple(args.prefixes))
