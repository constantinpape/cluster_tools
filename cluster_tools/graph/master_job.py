#! /usr/bin/python

import argparse
import subprocess
from wait_and_check import wait_and_check_single_job, wait_and_check_multiple_jobs
from wait_and_check import wait_and_check_multiple_jobnames


# TODO retrial for failed jobs
def master_job(n_jobs, n_scales):
    # submit jobs 1
    subprocess.call(['./jobs_step1.sh'])
    # wait for jobs 1
    failed_jobs = wait_and_check_multiple_jobs('graph_step1', n_jobs)
    if failed_jobs:
        print("Step 1 failed for following jobs:")
        print(failed_jobs)
        return

    # submit jobs 2 for scales 2:
    for scale in range(2, n_scales + 1):
        subprocess.call(['./jobs_step2_scale%i.sh' % scale])
        # wait for jobs 2
        failed_jobs = wait_and_check_multiple_jobs('graph_step2_scale%i' % scale, n_jobs)
        if failed_jobs:
            print("Step 2, scale %i failed for following jobs:" % scale)
            print(failed_jobs)
            return

    # submit jobs 3
    subprocess.call(['./jobs_step3.sh'])
    # wait for jobs 3
    failed_jobs = wait_and_check_single_job('graph_step3')
    if failed_jobs:
        print("Step 3 failed")
        return

    # submit jobs 4
    subprocess.call(['./jobs_step4.sh'])
    # wait for jobs 4
    failed_jobs = wait_and_check_multiple_jobnames(['graph_step4_scale%i'
                                                    for scale in range(1, n_scales + 1)], n_jobs)
    if failed_jobs:
        print("Step 4 failed for following jobs:")
        print(failed_jobs)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('n_scales', type=int)
    args = parser.parse_args()
    master_job(args.n_jobs, args.n_scales)
