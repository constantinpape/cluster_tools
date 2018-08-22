import json
from datetime import datetime
from subprocess import check_output

# Note that stdout is always piped to file, sowe can use it as logging
# TODO log-levels

def log(msg):
    print("%s: %s" % (str(datetime.now()), msg))


def log_block_success(block_id):
    print("%s: processed block %i" % (str(datetime.now()), block_id))


def log_job_success(job_id):
    print("%s: processed job %i" % (str(datetime.now()), job_id))


# woot, there is no native tail in python ???
def tail(path, n_lines):
    line_str = '-%i' % n_lines
    return check_output(['tail', line_str, path]).decode().split('\n')[:-1]
