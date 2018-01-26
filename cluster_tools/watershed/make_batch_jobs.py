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
