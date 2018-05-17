#! /usr/bin/python

import argparse
import subprocess
from wait_and_check import wait_and_check_single_job, wait_and_check_multiple_jobs


# TODO retrial for failed jobs
def master_job(n_jobs, have_ws_job):
    # submit jobs 0
    subprocess.call(['./jobs_step0.sh'])
    # wait for jobs 1
    failed_jobs = wait_and_check_single_job('dt_components_step0')
    if failed_jobs:
        print("Step 0 failed")
        return

    # submit jobs 1
    subprocess.call(['./jobs_step1.sh'])
    # wait for jobs 1
    failed_jobs = wait_and_check_multiple_jobs('dt_components_step1', n_jobs)
    if failed_jobs:
        print("Step 1 failed for following jobs:")
        print(failed_jobs)
        return

    # submit jobs 2
    subprocess.call(['./jobs_step2.sh'])
    # wait for jobs 2
    failed_jobs = wait_and_check_single_job('dt_components_step2')
    if failed_jobs:
        print("Step 2 failed")
        return

    # submit jobs 3
    subprocess.call(['./jobs_step3.sh'])
    # wait for jobs 3
    failed_jobs = wait_and_check_multiple_jobs('dt_components_step3', n_jobs)
    if failed_jobs:
        print("Step 3 failed for the following jobs:")
        print(failed_jobs)
        return

    # submit jobs 4
    subprocess.call(['./jobs_step4.sh'])
    # wait for jobs 4
    failed_jobs = wait_and_check_single_job('dt_components_step4')
    if failed_jobs:
        print("Step 4 failed")
        return

    # submit jobs 5
    subprocess.call(['./jobs_step5.sh'])
    # wait for jobs 5
    failed_jobs = wait_and_check_multiple_jobs('dt_components_step5', n_jobs)
    if failed_jobs:
        print("Step 5 failed for following jobs:")
        print(failed_jobs)
        return

    if have_ws_job:
        # submit jobs 6
        subprocess.call(['./jobs_step6.sh'])
        # wait for jobs 6
        failed_jobs = wait_and_check_multiple_jobs('dt_components_step6', n_jobs)
        if failed_jobs:
            print("Step 6 failed for following jobs:")
            print(failed_jobs)
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('have_ws_job', type=int)
    args = parser.parse_args()
    master_job(args.n_jobs, bool(args.have_ws_job))
