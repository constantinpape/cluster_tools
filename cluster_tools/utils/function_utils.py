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


# - global_config -> shebang, block_shape, roi_begin, roi_end
def load_global_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    shebang = config['shebang']
    block_shape = config['block_shape']
    roi_begin = config.get('roi_begin', None)
    roi_end = config.get('roi_end', None)
    return shebang, block_shape, roi_begin, roi_end


# - system_config -> threads_per_job, time_limit, mem_limit
def load_system_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    threads_per_job = config.get('threads_per_job', 1)
    # time limit in TODO which unit, minutes ?
    time_limit = config.get('time_limit', 60)
    # mem limit in TODO which unit, GB ?
    mem_limit = config.get('mem_limit', 1)
    return threads_per_job, time_limit, mem_limit


# woot, there is no native tail in python ???
def tail(path, n_lines):
    line_str = '-%i' % n_lines
    return check_output(['tail', line_str, path]).decode().split('\n')[:-1]
