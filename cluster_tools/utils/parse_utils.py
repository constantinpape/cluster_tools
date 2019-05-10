import os
import datetime
from subprocess import CalledProcessError

import numpy as np
from .function_utils import tail


################
# Parse runtimes
################


def parse_runtime(log_file):
    """ Parse the job run-time from a log-file
    """
    with open(log_file, 'r') as f:
        for ii, line in enumerate(f):
            if ii == 0:
                l0 = line
            l1 = line
    l0 = l0.strip("\n")
    l1 = l1.strip("\n")

    l0 = l0.split()[:2]
    l1 = l1.split()[:2]

    try:
        y0, m0, d0 = list(map(int, l0[0].split('-')))
        h0, min0, s0 = list(map(float, l0[1][:-1].split(':')))
    except ValueError as e:
        print(log_file)
        print(l0)
        raise e

    try:
        y1, m1, d1 = list(map(int, l1[0].split('-')))
        h1, min1, s1 = list(map(float, l1[1][:-1].split(':')))
    except ValueError as e:
        print(log_file)
        print(l1)
        raise e

    date0 = datetime.datetime(y0, m0, d0, int(h0), int(min0), int(s0))
    date1 = datetime.datetime(y1, m1, d1, int(h1), int(min1), int(s1))

    diff = (date1 - date0).total_seconds()
    return diff


def parse_runtime_task(log_prefix, max_jobs, return_summary=True):
    """ Parse all runtimes for jobs of a task and retrurn summary
    """
    runtimes = []
    for job_id in range(max_jobs):
        path = log_prefix + '%i.log' % job_id
        if not os.path.exists(path):
            break
        runtimes.append(parse_runtime(path))
    if return_summary:
        return (np.mean(runtimes), np.std(runtimes), len(runtimes))
    else:
        return runtimes


# TODO
def parse_runtime_segmentation_workflow():
    pass


######################
# Parse processed jobs
######################


def parse_job(log_file, job_id):
    """ Parse log file to check whether the corresponding
        job was finished successfully
    """
    # read the last line from the log file and check
    # whether it contains the "processed job" message
    try:
        last_line = tail(log_file, 1)[0]
    # if the file does not exist, this throws a `CalledProcessError`
    # if it does exist, but is empty, it throws a `IndexError`
    # in both cases, the job was unsuccessfull and we return False
    except (IndexError, CalledProcessError):
        return False

    # get rid of the datetime prefix and check
    msg = " ".join(last_line.split()[2:])
    return msg == "processed job %i" % job_id


# LSF appends its own logs to the out-file, so we need to parse differently
def parse_job_lsf(log_file, job_id):
    """ Parse lsf log file to check whether the corresponding
        job was finished successfully
    """
    with open(log_file, 'r') as f:
        for ll in f:
            ll = ll.rstrip()
            # '---------------' marks the begin of lsf log
            if ll.startswith('---------------'):
                return False
            try:
                # get rid of the datetime prefix and check
                msg = " ".join(ll.split()[2:])
            except Exception:
                # if this fails for some reason, there is something unexpected in
                # the log and we fail the job
                return False
            if msg == "processed job %i" % job_id:
                return True
    return False


########################
# Parse processed blocks
########################


def parse_blocks(log_file):
    """ Parse log file to return the blocks that were
        marked as processed
    """
    blocks = []
    with open(log_file, 'r') as f:
        for line in f:
            # get rid of date-time prefix
            line = ' '.join(line.split()[2:])
            # check if the line marks a block that has passed
            if line.startswith('processed block'):
                blocks.append(int(line.split()[-1]))
    return blocks


def parse_blocks_task(log_prefix, max_jobs, complete_job_list=[]):
    """ Parlse all processed blocks for jobs of a task
    """
    blocks = []
    for job_id in range(max_jobs):

        # don't need to parse if the job was marked as complete
        if job_id in complete_job_list:
            continue

        log_file = log_prefix + '%i.log' % job_id
        # log might not exist, even if this is not the last job
        if not os.path.exists(log_file):
            continue
        blocks.extend(parse_blocks(log_file))

    return blocks
