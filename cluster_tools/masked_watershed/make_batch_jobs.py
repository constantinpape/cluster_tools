import os
import stat
import fileinput
from shutil import copy, rmtree


https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python
def replace_shebang(file_path, shebang):
    for i, line in enumerate(fileinput.input(file_path, inplace=True)):
        if i == 0:
            print(shebang, end='')
        else:
            print(line, end='')


def make_executable(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)


def make_batch_jobs_step1(aff_path, aff_key,
                          mask_path, mask_key,
                          out_path, out_key, tmp_folder,
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

    copy(os.path.join(file_dir, 'implementation/1_watershed.py'), cwd)
    replace_shebang('1_watershed.py', shebang)
    make_executable('1_watershed.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        f.write('./0_prepare.py %s %s %s %s %s %s --tmp_folder %s --block_shape %s --chunks %s --n_jobs %s\n' %
                (aff_path, aff_key,
                 mask_path, mask_key,
                 out_path, out_key, tmp_folder,
                 ' '.join(map(str, block_shape)),
                 ' '.join(map(str, chunks)),
                 str(n_jobs)))
        # TODO we need to check for success here !

        for job_id in range(n_jobs):
            command = './1_watershed.py %s %s %s %s %s %s --tmp_folder %s --block_shape %s --block_file %s' % \
                      (aff_path, aff_key,
                       mask_path, mask_key,
                       out_path, out_key, tmp_folder,
                       ' '.join(map(str, block_shape)),
                       os.path.join(tmp_folder, '1_input_%i.npy' % job_id))
            if use_bsub:
                log_file = 'logs/log_masked_watershed_step1_%i.log' % job_id
                err_file = 'error_logs/err_masked_watershed_step1_%i.err' % job_id
                f.write('bsub -J masked_watershed_step1_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step2(out_path, out_key, tmp_folder,
                          block_shape, n_jobs, executable,
                          script_file='jobs_step2.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/2_get_offsets.py'), cwd)
    replace_shebang('2_get_offsets.py', shebang)
    make_executable('2_get_offsets.py')

    copy(os.path.join(file_dir, 'implementation/3_merge_blocks.py'), cwd)
    replace_shebang('3_merge_blocks.py', shebang)
    make_executable('3_merge_blocks.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        f.write('./2_get_offsets.py %s %s --tmp_folder %s --block_shape %s \n' %
                (out_path, out_key, tmp_folder,
                 ' '.join(map(str, block_shape))))

        for job_id in range(n_jobs):
            command = './3_merge_blocks.py %s %s' % \
                      (os.path.join(tmp_folder, '1_output_ovlps_%i.npy' % job_id), tmp_folder)
            if use_bsub:
                log_file = 'logs/log_masked_watershed_step2_%i.log' % job_id
                err_file = 'error_logs/err_masked_watershed_step2_%i.err' % job_id
                f.write('bsub -J masked_watershed_step2_%i -We %i -o %s -e %s \'%s\' \n' %
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

    copy(os.path.join(file_dir, 'implementation/4_ufd.py'), cwd)
    replace_shebang('4_ufd.py', shebang)
    make_executable('4_ufd.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './4_ufd.py %s %s' % (tmp_folder, str(n_jobs))
        if use_bsub:
            log_file = 'logs/log_masked_watershed_step3.log'
            err_file = 'error_logs/err_masked_watershed_step3.err'
            f.write('bsub -n %i -J masked_watershed_step3 -We %i -o %s -e %s \'%s\'\n' %
                    (n_threads, eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step4(out_path, out_key, tmp_folder,
                          block_shape, n_jobs, executable,
                          script_file='jobs_step4.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/5_write_ids.py'), cwd)
    replace_shebang('5_write_ids.py', shebang)
    make_executable('5_write_ids.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')

        for job_id in range(n_jobs):
            # --block_shape %s' % \
            command = './5_write_ids.py %s %s %s %s --block_shape %s' % \
                (out_path, out_key, tmp_folder,
                 os.path.join(tmp_folder, '1_input_%i.npy' % job_id),
                 ' '.join(map(str, block_shape)))
            if use_bsub:
                log_file = 'logs/log_masked_watershed_step4_%i.log' % job_id
                err_file = 'error_logs/err_masked_watershed_step4_%i.err' % job_id
                f.write('bsub -J masked_watershed_step4_%i -We %i -o %s -e %s \'%s\' \n' %
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


def make_batch_jobs(aff_path, aff_key,
                    mask_path, mask_key,
                    out_path, out_key, tmp_folder,
                    block_shape, chunks, n_jobs, executable,
                    eta=15, n_threads_ufd=1, use_bsub=True):

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

    make_batch_jobs_step1(aff_path, aff_key,
                          mask_path, mask_key,
                          out_path, out_key, tmp_folder,
                          block_shape, chunks, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta_[0])

    make_batch_jobs_step2(out_path, out_key, tmp_folder, block_shape,  n_jobs,
                          executable, use_bsub=use_bsub, eta=eta_[1])

    make_batch_jobs_step3(tmp_folder, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta_[2], n_threads=n_threads_ufd)

    make_batch_jobs_step4(out_path, out_key, tmp_folder, block_shape, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta_[3])

    make_master_job(n_jobs, executable, 'master.sh')
