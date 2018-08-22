#! /usr/bin/python

import time
import argparse
import subprocess
from wait_and_check import wait_and_check_multiple_jobs, wait_and_check_single_job


# TODO retrial for failed jobs
def master_job(n_jobs, with_rf):

    t_tot = time.time()
    # submit jobs 1
    subprocess.call(['./jobs_step1.sh'])
    # wait for jobs 1, only if we have a random forest
    if with_rf:
        failed_jobs = wait_and_check_multiple_jobs('costs_step1', n_jobs)
        if failed_jobs:
            print("Step 1 failed for following jobs:")
            print(failed_jobs)
            return

    # submit jobs 2
    subprocess.call(['./jobs_step2.sh'])
    # wait for jobs 3
    failed_jobs = wait_and_check_single_job('costs_step2')
    if failed_jobs:
        print("Step 2 failed")
        return

    t_tot = time.time() - t_tot
    print("All jobs finished successfully in %f s" % t_tot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('with_rf', type=int)
    args = parser.parse_args()
    master_job(args.n_jobs, bool(args.with_rf))
