import os
import stat
import fileinput
from shutil import copy, rmtree


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


def make_log_dirs(tmp_folder):
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
