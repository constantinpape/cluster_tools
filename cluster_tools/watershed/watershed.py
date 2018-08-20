#! /bin/python

import sys
import json

import vigra
import nifty.tools as nt

# TODO which task do we need
from cluster_tools.cluster_tasks import x
from cluster_tools.volume_util import file_reader, volume_util, get_shape
from cluster_tools.functional_api import log_job_success, log_block_success, log


#
# Base Watershed Task
#

# TODO
class Watershed():
    pass


#
# Mixins for different implementations
#

class WatershedLocal():
    pass


class WatershedSlurm():
    pass


#
# Implementation
#

def _ws_block(blocking, block_id):
    pass


def watershed(job_id, config_path):
    log("start job %i" % job_id)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']
    shape = list(get_shape(input_path, input_key))
    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    # read the output config
    output_path = config['output_path']
    output_key = config['output_key']

    # get the blocking
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    # submit blocks
    for block_id in block_list:
        _ws_block(blocking, block_id)
    # log success
    log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(config_path.split('.')[0].split('_')[-1])
    watershed(job_id, config_path)
