from datetime import datetime


# stdout is always piped to file, so we can use it as logging
def log(msg):
    print("%s: %s" % (str(datetime.now()), msg))


def log_block_success(block_id):
    print("%s: processed block %i" % (str(datetime.now()), block_id))


def log_job_success(job_id):
    print("%s: processed job %i" % (str(datetime.now()), job_id))


# pythonic implementation of
# tail -<n_lines> <path>
def tail(path, n_lines):
    out = []
    with open(path, "r") as f:
        for i, line in enumerate(reversed(list(f)), 1):
            out.append(line.rstrip())
            if i == n_lines:
                break
    return out[::-1]
