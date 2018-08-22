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


def make_batch_jobs_step1(in_path, in_key, out_path, out_key, tmp_folder,
                          block_shape, chunks, n_jobs, executable,
                          script_file='jobs_step1.sh', use_bsub=True, eta=5):

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

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        f.write('./0_prepare.py %s %s %s %s --tmp_folder %s --block_shape %s --chunks %s --n_jobs %s\n' %
                (in_path, in_key, out_path, out_key, tmp_folder,
                 ' '.join(map(str, block_shape)),
                 ' '.join(map(str, chunks)),
                 str(n_jobs)))

        for job_id in range(n_jobs):
            command = './1_blockwise_cc.py %s %s %s %s --tmp_folder %s --block_shape %s --block_file %s' % \
                      (in_path, in_key, out_path, out_key, tmp_folder,
                       ' '.join(map(str, block_shape)),
                       os.path.join(tmp_folder, '1_input_%i.npy' % job_id))
            if use_bsub:
                log_file = 'logs/log_cc_ufd_step1_%i.log' % job_id
                err_file = 'error_logs/err_cc_ufd_step1_%i.err' % job_id
                f.write('bsub -J cc_ufd_step1_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step2(tmp_folder, n_jobs, executable,
                          script_file='jobs_step2.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/2_process_overlaps.py'), cwd)
    replace_shebang('2_process_overlaps.py', shebang)
    make_executable('2_process_overlaps.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        for job_id in range(n_jobs):
            command = './2_process_overlaps.py %s %s \n' % \
                      (tmp_folder, os.path.join(tmp_folder, '1_output_ovlps_%i.npy' % job_id))
            if use_bsub:
                log_file = 'logs/log_cc_ufd_step2_%i.log' % job_id
                err_file = 'error_logs/err_cc_ufd_step2_%i.err' % job_id
                f.write('bsub -J cc_ufd_step2_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step3(tmp_folder, n_jobs, executable,
                          script_file='jobs_step3.sh', use_bsub=True, eta=5, n_threads=1):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/3_ufd.py'), cwd)
    replace_shebang('3_ufd.py', shebang)
    make_executable('3_ufd.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './3_ufd.py %s %s \n' % (tmp_folder, str(n_jobs))
        if use_bsub:
            log_file = 'logs/log_cc_ufd_step3.log'
            err_file = 'error_logs/err_cc_ufd_step3.err'
            f.write('bsub -n %i -J cc_ufd_step3 -We %i -o %s -e %s \'%s\'\n' %
                    (n_threads, eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step4(out_path, out_key, tmp_folder, block_shape,
                          n_jobs, executable, script_file='jobs_step4.sh',
                          use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/4_write.py'), cwd)
    replace_shebang('4_write.py', shebang)
    make_executable('4_write.py')

    with open(script_file, 'w') as f:

        f.write('#! /bin/bash\n')
        for job_id in range(n_jobs):
            command = './4_write.py %s %s %s %s --block_shape %s \n' % \
                       (os.path.join(tmp_folder, '1_input_%i.npy' % job_id),
                        out_path, out_key, tmp_folder,
                        ' '.join(map(str, block_shape)))
            if use_bsub:
                log_file = 'logs/log_cc_ufd_step4_%i.log' % job_id
                err_file = 'error_logs/err_cc_ufd_step4_%i.err' % job_id
                f.write('bsub -J cc_ufd_step4_%i -We %i -o %s -e %s \'%s\'\n' %
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


def make_batch_jobs(in_path, in_key, out_path, out_key, tmp_folder,
                    block_shape, chunks, n_jobs, executable,
                    eta=5, n_threads_ufd=1, use_bsub=True):

    assert isinstance(eta, (int, list, tuple))
    if isinstance(eta, (list, tuple)):
        assert len(eta) == 4
        assert all(isinstance(ee, int) for ee in eta)
        eta_ = eta
    else:
        eta_ = (eta,) * 4

    # clean logs
    if os.path.exists('error_logs'):
        rmtree('error_logs')
    os.mkdir('error_logs')

    if os.path.exists('logs'):
        rmtree('logs')
    os.mkdir('logs')

    make_batch_jobs_step1(in_path, in_key, out_path, out_key, tmp_folder,
                          block_shape, chunks, n_jobs, executable,
                          script_file='jobs_step1.sh', use_bsub=use_bsub, eta=eta_[0])

    make_batch_jobs_step2(tmp_folder, n_jobs, executable,
                          script_file='jobs_step2.sh', use_bsub=use_bsub, eta=eta_[1])

    make_batch_jobs_step3(tmp_folder, n_jobs, executable,
                          script_file='jobs_step3.sh', use_bsub=use_bsub,
                          eta=eta_[2], n_threads=n_threads_ufd)

    make_batch_jobs_step4(out_path, out_key, tmp_folder, block_shape,
                          n_jobs, executable,
                          script_file='jobs_step4.sh', use_bsub=use_bsub, eta=eta_[3])

    make_master_job(n_jobs, executable, 'master.sh')
