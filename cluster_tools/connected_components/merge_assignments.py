#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import vigra
import nifty.ufd as nufd

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Find Labeling Tasks
#

class MergeAssignmentsBase(luigi.Task):
    """ MergeAssignments base class
    """

    task_name = "merge_assignments"
    src_file = os.path.abspath(__file__)
    allow_retry = False

    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    shape = luigi.ListParameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        block_list = vu.blocks_in_volume(self.shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        config = self.get_task_config()
        config.update({"output_path": self.output_path, "output_key": self.output_key,
                       "tmp_folder": self.tmp_folder, "n_jobs": n_jobs, "block_list": block_list})

        # we only have a single job to find the labeling
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(1)


class MergeAssignmentsLocal(MergeAssignmentsBase, LocalTask):
    """
    MergeAssignments on local machine
    """
    pass


class MergeAssignmentsSlurm(MergeAssignmentsBase, SlurmTask):
    """
    MergeAssignments on slurm cluster
    """
    pass


class MergeAssignmentsLSF(MergeAssignmentsBase, LSFTask):
    """
    MergeAssignments on lsf cluster
    """
    pass


def merge_assignments(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, "r") as f:
        config = json.load(f)
    output_path = config["output_path"]
    output_key = config["output_key"]

    tmp_folder = config["tmp_folder"]
    n_jobs = config["n_jobs"]
    block_list = config["block_list"]

    id_prefix = "ids"
    assignment_prefix = "cc_assignments"

    # load labels
    label_paths = [os.path.join(tmp_folder, f"{id_prefix}_{block_id}.npy") for block_id in block_list]
    labels = [np.load(pp) if os.path.exists(pp) else [0] for pp in label_paths]
    labels = np.unique(np.concatenate(labels))

    # load assignments
    assignment_paths = [os.path.join(tmp_folder, f"{assignment_prefix}_{job_id}.npy") for job_id in range(n_jobs)]
    assignments = [np.load(pp) for pp in assignment_paths if os.path.exists(pp)]

    if assignments:
        assignments = np.concatenate(assignments, axis=0)
        assignments = np.unique(assignments, axis=0)
        assert assignments.shape[1] == 2
        fu.log("have %i pairs of node assignments" % len(assignments))
        have_assignments = True
    else:
        fu.log("did not find any node assignments and will not merge any components")
        have_assignments = False

    if have_assignments:
        n_elements = int(labels.max()) + 1
        ufd = nufd.ufd(n_elements)
        ufd.merge(assignments)
        label_assignments = ufd.find(labels)
    else:
        label_assignemnts = labels.copy()

    n_labels = len(labels)
    label_assignemnts, max_id, _ = vigra.analysis.relabelConsecutive(label_assignments, keep_zeros=True, start_label=1)
    assert len(label_assignments) == n_labels
    fu.log("reducing the number of labels from %i to %i" % (n_labels, max_id + 1))

    label_assignments = np.concatenate([labels[:, None], label_assignments[:, None]], axis=1).astype("uint64")
    chunks = (min(65334, n_labels), 2)
    with vu.file_reader(output_path) as f:
        f.create_dataset(output_key, data=label_assignments, compression="gzip", chunks=chunks)
    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    merge_assignments(job_id, path)
