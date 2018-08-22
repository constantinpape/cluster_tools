import os
import stat
import fileinput
from shutil import copy, rmtree

import luigi
import numpy as np
import z5py


# https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python
def replace_shebang(file_path, shebang):
    for i, line in enumerate(fileinput.input(file_path, inplace=True)):
        if i == 0:
            print(shebang, end='')
        else:
            print(line, end='')


def make_executable(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def copy_and_replace(src, dest,
                     shebang='#! /groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'):
    copy(src, dest)
    replace_shebang(dest, shebang)
    make_executable(dest)


def make_dirs(tmp_folder):
    # make the tmpdir
    try:
        os.mkdir(tmp_folder)
    except OSError:
        pass
    # make log and err dir
    log_dir = os.path.join(tmp_folder, 'logs')
    err_dir = os.path.join(tmp_folder, 'error_logs')
    try:
        os.mkdir(log_dir)
    except OSError:
        pass
    try:
        os.mkdir(err_dir)
    except OSError:
        pass


# Dummy task that is always fullfilled
class DummyTask(luigi.Task):
    def output(self):
        return luigi.LocalTarget('.')


def normalize_and_save_assignments(path, key, assignments, n_threads=1,
                                   offset_segment_labels=True):

    assert assignments.ndim == 1
    n_nodes = len(assignments)
    # save as look-up table from fragment (node) id
    # to segment id
    # note that the node ids here are dense
    # we offset the segment ids by the node ids, to use the same id-space
    # for both fragment (node) and segment ids
    # (except for 0, which is still mapped to 0)
    if offset_segment_labels:
        assignments[1:] += n_nodes
    out = np.dstack([np.arange(n_nodes, dtype='uint64'), assignments])

    # to save as row vectors instead of columvectors:
    # out = np.vstack([np.arange(n_nodes, dtype='uint64'),
    #                  assignments + n_nodes])

    f_out = z5py.File(path)
    node_shape = (n_nodes, 2)
    # 64 ** 2 chunks
    chunks = (min(n_nodes, 262214), 1)
    ds_nodes = f_out.require_dataset(key, dtype='uint64',
                                     shape=node_shape, chunks=chunks)
    ds_nodes.n_threads = n_threads
    ds_nodes[:] = out
