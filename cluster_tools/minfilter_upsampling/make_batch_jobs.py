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


def make_batch_jobs_step1(mask_path, mask_key,
                          ds_mask_path, ds_mask_key,
                          out_path, out_key,
                          tmp_folder, chunks,
                          filter_shape, block_shape,
                          sampling_factor,
                          n_jobs, executable,
                          script_file='jobs_step1.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    make_executable('0_prepare.py')

    copy(os.path.join(file_dir, 'implementation/1_minfilter.py'), cwd)
    replace_shebang('1_minfilter.py', shebang)
    make_executable('1_minfilter.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        f.write('./0_prepare.py %s %s %s %s %s %s --tmp_folder %s --chunks %s --block_shape %s --sampling_factor %s --n_jobs %s\n' %
                (mask_path, mask_key,
                 ds_mask_path, ds_mask_key,
                 out_path, out_key, tmp_folder,
                 ' '.join(map(str, chunks)),
                 ' '.join(map(str, block_shape)),
                 ' '.join(map(str, sampling_factor)),
                 str(n_jobs)))
        # TODO we need to check for success here !

        for job_id in range(n_jobs):
            command = './1_minfilter.py %s %s %s %s %s %s --filter_shape %s --block_shape %s --sampling_factor %s --block_file %s' % \
                      (mask_path, mask_key,
                       ds_mask_path, ds_mask_key,
                       out_path, out_key,
                       ' '.join(map(str, filter_shape)),
                       ' '.join(map(str, block_shape)),
                       ' '.join(map(str, sampling_factor)),
                       os.path.join(tmp_folder, '1_input_%i.npy' % job_id))
            if use_bsub:
                log_file = 'logs/log_minfilter_upsampling_step1_%i.log' % job_id
                err_file = 'error_logs/err_minfilter_upsampling_step1_%i.err' % job_id
                f.write('bsub -J minfilter_step1_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_master_job(n_jobs, executable, script_file):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'master_job.py'), cwd)
    replace_shebang('master_job.py', shebang)
    make_executable('master_job.py')

    parent_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
    copy(os.path.join(parent_dir, 'wait_and_check.py'), cwd)

    with open(script_file, 'w') as f:
        f.write('./master_job.py %i\n' % n_jobs)
    make_executable(script_file)


def make_batch_jobs(mask_path, mask_key,
                    out_path, out_key,
                    chunks, filter_shape,
                    block_shape, tmp_folder,
                    n_jobs, executable,
                    eta=5, use_bsub=True):

    assert isinstance(eta, int)

    # clean logs
    if os.path.exists('error_logs'):
        rmtree('error_logs')
    os.mkdir('error_logs')

    if os.path.exists('logs'):
        rmtree('logs')
    os.mkdir('logs')

    make_batch_jobs_step1(mask_path, mask_key,
                          out_path, out_key,
                          tmp_folder, chunks, filter_shape,
                          block_shape, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta)

    make_master_job(n_jobs, executable, 'master.sh')
