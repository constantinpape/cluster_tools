#! /usr/bin/python

import time
import argparse
import subprocess
from wait_and_check import wait_and_check_single_job, wait_and_check_multiple_jobs


# TODO retrial for failed jobs
def master_job(n_jobs, n_scales):

    t_tot = time.time()
    # submit jobs 1
    subprocess.call(['./jobs_step1.sh'])
    # wait for jobs 1
    failed_jobs = wait_and_check_single_job('multicut_step1', n_jobs)
    if failed_jobs:
        print("Step 1 failed")
        return

    # submit jobs 2 for
    for scale in range(n_scales):
        subprocess.call(['./jobs_step2subproblem_scale%i.sh' % scale])
        # wait for subproblem extractions
        failed_jobs = wait_and_check_multiple_jobs('multicut_step2_scale%i' % scale, n_jobs)
        if failed_jobs:
            print("Step 2, scale %i, subproblem extraction failed for following jobs:" % scale)
            print(failed_jobs)
            return
        # schedule and wait for reduced problem
        subprocess.call(['./jobs_step2reduce_scale%i.sh' % scale])
        failed_jobs = wait_and_check_single_job('multicut_step2_scale%i' % scale)
        if failed_jobs:
            print("Step 2 failed, scale %i, reduce failed" % scale)
            return

    # submit jobs 3
    subprocess.call(['./jobs_step3.sh'])
    # wait for jobs 3
    failed_jobs = wait_and_check_single_job('multicut_step3')
    if failed_jobs:
        print("Step 3 failed")
        return

    t_tot = time.time() - t_tot
    print("All jobs finished successfully in %f s" % t_tot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('n_scales', type=int)
    args = parser.parse_args()
    master_job(args.n_jobs, args.n_scales)
