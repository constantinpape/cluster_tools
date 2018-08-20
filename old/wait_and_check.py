import subprocess
import os
import time


def wait_for_jobs(user, max_wait_time=None):
    t_start = time.time()
    while True:
        time.sleep(5)
        n_running = subprocess.check_output(['bjobs | grep %s | wc -l' % user], shell=True).decode()
        n_running = int(n_running.strip('\n'))
        if n_running == 0:
            break
        if max_wait_time is not None:
            t_wait = time.time() - t_start
            if t_wait > max_wait_time:
                print("MAX WAIT TIME EXCEEDED")
                break


def wait_and_check_multiple_jobs(job_prefix, n_jobs, user='papec'):

    success_marker = 'Success'
    wait_for_jobs(user)

    jobs_failed = []
    for job_id in range(n_jobs):
        log_file = './logs/log_%s_%i.log' % (job_prefix, job_id)

        have_log = os.path.exists(log_file)
        if not have_log:
            jobs_failed.append(job_id)
            continue

        with open(log_file, 'r') as f:
            out = f.readline()
            have_success = out[:len(success_marker)] == success_marker
        if not have_success:
            jobs_failed.append(job_id)
            continue

    return jobs_failed


def wait_and_check_single_job(job_name, user='papec'):

    success_marker = 'Success'
    wait_for_jobs(user)

    log_file = './logs/log_%s.log' % job_name

    job_failed = False
    have_log = os.path.exists(log_file)
    if not have_log:
        job_failed = True

    with open(log_file, 'r') as f:
        out = f.readline()
        have_success = out[:len(success_marker)] == success_marker
    if not have_success:
        job_failed = True

    return job_failed


def wait_and_check_multiple_jobnames(job_names, user='papec'):

    success_marker = 'Success'
    wait_for_jobs(user)

    jobs_failed = []
    for job_name in job_names:
        log_file = './logs/log_%s.log' % job_name

        have_log = os.path.exists(log_file)
        if not have_log:
            jobs_failed.append(job_name)
            continue

        with open(log_file, 'r') as f:
            out = f.readline()
            have_success = out[:len(success_marker)] == success_marker
        if not have_success:
            jobs_failed.append(job_name)
            continue

    return jobs_failed
