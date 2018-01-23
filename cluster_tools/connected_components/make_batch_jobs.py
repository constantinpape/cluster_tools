import os
import fileinput
from shutil import copy


# https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python
def replace_shebang(file_path, shebang):
    for i, line in enumerate(fileinput.input(file_path, inplace=True)):
        if i == 0:
            print(shebang, end='')
        else:
            print(line, end='')


def make_batch_jobs_step1(in_path, in_key, out_path, out_key, tmp_folder,
                          block_shape, chunks, n_jobs, executable,
                          script_file='jobs_step1.sh', use_bsub=True):

    # TODO make sure that chunks and block_shape match

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))[0]
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    copy(os.path.join(file_dir, 'implementation/1_blockwise_cc.py'), cwd)

    with open(script_file, 'w') as f:
        f.write('0_prepare.py %s %s %s %s --tmp_folder %s --block_shape %s --chunks %s --n_jobs %s\n' %
                (in_path, in_key, out_path, out_key, tmp_folder,
                 ''.join(map(str, block_shape)),
                 ''.join(map(str, chunks)),
                 str(n_jobs)))

        for job_id in range(n_jobs):
            if use_bsub:
                # TODO
                pass
            else:
                f.write('1_blockwise_cc.py %s %s %s %s --block_shape %s --block_file %s\n' %
                        (in_path, in_key, out_path, out_key, tmp_folder,
                         ''.join(map(str, block_shape)),
                         os.path.join(tmp_folder, '1_inputs_%i.npy' % job_id)))
