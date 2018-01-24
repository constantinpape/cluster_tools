import os
import stat
import fileinput
from shutil import copy


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


def make_batch_jobs_step1(in_path, in_key, out_path, out_key, tmp_folder,
                          block_shape, chunks, n_jobs, executable,
                          script_file='jobs_step1.sh', use_bsub=True):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    make_executable('0_prepare.py')

    copy(os.path.join(file_dir, 'implementation/1_blockwise_cc.py'), cwd)
    replace_shebang('1_blockwise_cc.py', shebang)
    make_executable('1_blockwise_cc.py')

    copy(os.path.join(file_dir, 'implementation/1a_merge_ovlp_ids.py'), cwd)
    replace_shebang('1a_merge_ovlp_ids.py', shebang)
    make_executable('1a_merge_ovlp_ids.py')

    with open(script_file, 'w') as f:
        f.write('./0_prepare.py %s %s %s %s --tmp_folder %s --block_shape %s --chunks %s --n_jobs %s\n' %
                (in_path, in_key, out_path, out_key, tmp_folder,
                 ' '.join(map(str, block_shape)),
                 ' '.join(map(str, chunks)),
                 str(n_jobs)))

        for job_id in range(n_jobs):
            if use_bsub:
                # TODO
                pass
            else:
                f.write('./1_blockwise_cc.py %s %s %s %s --tmp_folder %s --block_shape %s --block_file %s\n' %
                        (in_path, in_key, out_path, out_key, tmp_folder,
                         ' '.join(map(str, block_shape)),
                         os.path.join(tmp_folder, '1_input_%i.npy' % job_id)))

        f.write('./1a_merge_ovlp_ids.py %s %s' % (tmp_folder, str(n_jobs)))

    make_executable(script_file)


def make_batch_jobs_step2():
    pass
