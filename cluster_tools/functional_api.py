from datetime import datetime

# Note that stdout is always piped to file, sowe can use it as logging
# TODO log-levels

def log(msg):
    print("%s: %s" % (str(datetime.now()), msg))


def log_block_success(block_id):
    print("%s: processed block %i" % (str(datetime.now()), block_id))


def log_job_success(job_id):
    print("%s: processed job %i" % (str(datetime.now()), job_id))
