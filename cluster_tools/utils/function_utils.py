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


# TODO load for different parts of the global config
def load_global_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    shebang = config['shebang']
    block_shape = config['block_shape']
    roi_begin = config.get('roi_begin', None)
    roi_end = config.get('roi_end', None)
    return shebang, block_shape, roi_begin, roi_end


# woot, there is no native tail in python ???
def tail(path, n_lines):
    line_str = '-%i' % n_lines
    return check_output(['tail', line_str, path]).decode().split('\n')[:-1]
