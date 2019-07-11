#! /bin/python

import os
import sys
import json
import nifty.tools as nt

# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
from cluster_tools.utils.numpy_utils import set_numpy_threads
set_numpy_threads(1)
import numpy as np

from cluster_tools.utils import volume_utils as vu


#
# Validation measure tasks
#

class ObjectViBase(luigi.Task):
    """ ObjectVi base class
    """

    task_name = 'object_vi'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    seg_path = luigi.Parameter()
    seg_key = luigi.Parameter()
    gt_path = luigi.Parameter()
    gt_key = luigi.Parameter()
    morpho_path = luigi.Parameter()
    morpho_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)
        chunk_size = 500

        # load the task config
        config = self.get_task_config()
        config.update({'seg_path': self.seg_path, 'seg_key': self.seg_key,
                       'gt_path': self.gt_path, 'gt_key': self.gt_key,
                       'morpho_path': self.morpho_path, 'morpho_key': self.morpho_key,
                       'chunk_size': chunk_size, 'tmp_folder': self.tmp_folder})
        with vu.file_reader(self.seg_path) as f:
            n_labels = int(f[self.seg_key].attrs['maxId']) + 1

        block_list = vu.blocks_in_volume([n_labels], [chunk_size])
        n_jobs = min(self.max_jobs, len(block_list))
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class ObjectViLocal(ObjectViBase, LocalTask):
    """ ObjectVi on local machine
    """
    pass


class ObjectViSlurm(ObjectViBase, SlurmTask):
    """ ObjectVi on slurm cluster
    """
    pass


class ObjectViLSF(ObjectViBase, LSFTask):
    """ ObjectVi on lsf cluster
    """
    pass


# TODO
def compute_object_id(seg, gt, label_id):
    pass


def object_vis_for_label_range(blocking, block_id,
                               ds_seg, ds_gt, morphology):
    block = blocking.getBlock(block_id)
    id_a, id_b = block.blockBegin, block.blockEnd

    # TODO
    bbs = ''

    results = [compute_object_id(ds_seg[bbs[label_id]], ds_gt[bbs[label_id]], label_id)
               for label_id in range(id_a, id_b)]
    return results


def object_vi(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    seg_path = config['seg_path']
    seg_key = config['seg_key']
    gt_path = config['gt_path']
    gt_key = config['gt_key']
    morpho_path = config['morpho_path']
    morpho_key = config['morpho_key']

    chunk_size = config['chunk_size']
    tmp_folder = config['tmp_folder']
    block_list = config['block_list']

    with vu.file_reader(morphology, 'r') as f:
        morphology = f[morpho_key][:]

    with vu.file_reader(seg_path, 'r') as f_seg, vu.file_reader(gt_path, 'r') as f_gt:
        ds_seg = f_seg[seg_key]
        ds_gt = f_gt[gt_key]

        n_labels = int(ds_seg.attrs['maxId']) + 1
        blocking = nt.blocking([0], [n_labels], [chunk_size])

        results = [object_vis_for_label_range(blocking, block_id,
                                              ds_seg, ds_gt,
                                              morphology)
                   for block_id in range(blocking.numberOfBlocks)]

    results = {k: v for res in results for k, v in res.items()}
    output_path = os.path.join(tmp_folder, 'object_vis_%i.json' % job_id)
    with open(output_path, 'w') as f:
        json.dump(results, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    object_vi(job_id, path)
