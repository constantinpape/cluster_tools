#! /usr/bin/python

import time
import argparse
import subprocess
from wait_and_check import wait_and_check_multiple_jobs


# TODO retrial for failed jobs
def master_job(n_jobs):

    t_tot = time.time()
    # submit jobs 1
    subprocess.call(['./jobs_step1.sh'])
    # wait for jobs 1
    failed_jobs = wait_and_check_multiple_jobs('minfilter_step1', n_jobs)
    if failed_jobs:
        print("Step 1 failed for following jobs:")
        print(failed_jobs)
        return

    t_tot = time.time() - t_tot
    print("All jobs ran successfully in %f s" % t_tot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    master_job(args.n_jobs)
