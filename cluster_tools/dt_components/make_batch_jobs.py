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


def make_batch_jobs_step0(path, mask_key, out_key, cache_folder,
                          n_jobs, block_shape, executable, ws_key,
                          script_file='jobs_step0.sh',
                          use_bsub=True, eta=5):
    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step0_prepare.py'), cwd)
    replace_shebang('step0_prepare.py', shebang)
    make_executable('step0_prepare.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        if ws_key is None:
            command = './step0_prepare.py %s %s %s %s %i --block_shape %s\n' % \
                (path, mask_key, out_key,
                 cache_folder, n_jobs,
                 ' '.join(map(str, block_shape)))
        else:
            command = './step0_prepare.py %s %s %s %s %i --block_shape %s --ws_key %s\n' % \
                (path, mask_key, out_key,
                 cache_folder, n_jobs,
                 ' '.join(map(str, block_shape)), ws_key)
        if use_bsub:
            log_file = 'logs/log_dt_components_step0.log'
            err_file = 'error_logs/err_dt_components_step0.err'
            f.write('bsub -J dt_components_step0 -We %i -o %s -e %s \'%s\' \n' %
                    (eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step1(path, aff_key, out_key, mask_key,
                          cache_folder, n_jobs, executable,
                          script_file='jobs_step1.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step1_dt.py'), cwd)
    replace_shebang('step1_dt.py', shebang)
    make_executable('step1_dt.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')

        # for job_id in [38]:
        for job_id in range(n_jobs):
            command = './step1_dt.py %s %s %s %s %s %i' % (path, aff_key, out_key,
                                                           mask_key, cache_folder, job_id)
            if use_bsub:
                log_file = 'logs/log_dt_components_step1_%i.log' % job_id
                err_file = 'error_logs/err_dt_components_step1_%i.err' % job_id
                f.write('bsub -J dt_components_step1_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step2(cache_folder, n_jobs, executable,
                          script_file='jobs_step2.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step2_offsets.py'), cwd)
    replace_shebang('step2_offsets.py', shebang)
    make_executable('step2_offsets.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './step2_offsets.py %s %i' % (cache_folder, n_jobs)
        if use_bsub:
            log_file = 'logs/log_dt_components_step2.log'
            err_file = 'error_logs/err_dt_components_step2.err'
            f.write('bsub -J dt_components_step2 -We %i -o %s -e %s \'%s\' \n' %
                    (eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step3(path, out_key, cache_folder, n_jobs, executable,
                          script_file='jobs_step3.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step3_merge_components.py'), cwd)
    replace_shebang('step3_merge_components.py', shebang)
    make_executable('step3_merge_components.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')

        # for job_id in [38]:
        for job_id in range(n_jobs):
            command = './step3_merge_components.py %s %s %s %i' % (path, out_key,
                                                                   cache_folder, job_id)
            if use_bsub:
                log_file = 'logs/log_dt_components_step3_%i.log' % job_id
                err_file = 'error_logs/err_dt_components_step3_%i.err' % job_id
                f.write('bsub -J dt_components_step3_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step4(cache_folder, n_jobs, executable, n_threads,
                          script_file='jobs_step4.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step4_ufd.py'), cwd)
    replace_shebang('step4_ufd.py', shebang)
    make_executable('step4_ufd.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './step4_ufd.py %s %i' % (cache_folder, n_jobs)
        if use_bsub:
            log_file = 'logs/log_dt_components_step4.log'
            err_file = 'error_logs/err_dt_components_step4.err'
            f.write('bsub -n %i -J dt_components_step4 -We %i -o %s -e %s \'%s\' \n' %
                    (n_threads, eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step5(path, out_key, cache_folder, n_jobs, executable,
                          script_file='jobs_step5.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step5_write.py'), cwd)
    replace_shebang('step5_write.py', shebang)
    make_executable('step5_write.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')

        # for job_id in [38]:
        for job_id in range(n_jobs):
            command = './step5_write.py %s %s %s %i' % (path, out_key,
                                                        cache_folder, job_id)
            if use_bsub:
                log_file = 'logs/log_dt_components_step5_%i.log' % job_id
                err_file = 'error_logs/err_dt_components_step5_%i.err' % job_id
                f.write('bsub -J dt_components_step5_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step6(path, aff_key, seed_key, mask_key, out_key,
                          cache_folder, n_jobs, executable,
                          script_file='jobs_step6.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step6_watershed.py'), cwd)
    replace_shebang('step6_watershed.py', shebang)
    make_executable('step6_watershed.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')

        # for job_id in [38]:
        for job_id in range(n_jobs):
            command = './step6_watershed.py %s %s %s %s %s %s %i' % (path, aff_key, seed_key,
                                                                     mask_key, out_key,
                                                                     cache_folder, job_id)
            if use_bsub:
                log_file = 'logs/log_dt_components_step6_%i.log' % job_id
                err_file = 'error_logs/err_dt_components_step6_%i.err' % job_id
                f.write('bsub -J dt_components_step6_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step7(cache_folder, n_jobs, executable,
                          script_file='jobs_step7.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step7_offsets_max_seeds.py'), cwd)
    replace_shebang('step7_offsets_max_seeds.py', shebang)
    make_executable('step7_offsets_max_seeds.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './step7_offsets_max_seeds.py %s %i' % (cache_folder, n_jobs)
        if use_bsub:
            log_file = 'logs/log_dt_components_step7.log'
            err_file = 'error_logs/err_dt_components_step7.err'
            f.write('bsub -J dt_components_step7 -We %i -o %s -e %s \'%s\' \n' %
                    (eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step8(path, out_key, cache_folder, n_jobs, executable,
                          script_file='jobs_step8.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/step8_write_max_offsets.py'), cwd)
    replace_shebang('step8_write_max_offsets.py', shebang)
    make_executable('step8_write_max_offsets.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')

        # for job_id in [38]:
        for job_id in range(n_jobs):
            command = './step8_write_max_offsets.py %s %s %s %i' % (path, out_key,
                                                                    cache_folder, job_id)
            if use_bsub:
                log_file = 'logs/log_dt_components_step8_%i.log' % job_id
                err_file = 'error_logs/err_dt_components_step8_%i.err' % job_id
                f.write('bsub -J dt_components_step8_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_master_job(n_jobs, executable, script_file, have_ws_job):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'master_job.py'), cwd)
    replace_shebang('master_job.py', shebang)
    make_executable('master_job.py')

    parent_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
    copy(os.path.join(parent_dir, 'wait_and_check.py'), cwd)

    ws_job = 1 if have_ws_job else 0
    with open(script_file, 'w') as f:
        f.write('./master_job.py %i %i\n' % (n_jobs, ws_job))
    make_executable(script_file)


def make_batch_jobs(path, aff_key, out_key, mask_key,
                    cache_folder, n_jobs, block_shape, executable,
                    ws_key=None,
                    eta=15, n_threads_ufd=1, use_bsub=True):

    assert isinstance(eta, (int, list, tuple))
    if isinstance(eta, (list, tuple)):
        assert len(eta) == 9
        assert all(isinstance(ee, int) for ee in eta)
        eta_ = eta
    else:
        eta_ = (eta,) * 9

    # clean logs
    if os.path.exists('error_logs'):
        rmtree('error_logs')
    os.mkdir('error_logs')

    if os.path.exists('logs'):
        rmtree('logs')
    os.mkdir('logs')

    have_ws_job = ws_key is not None

    make_batch_jobs_step0(path, mask_key, out_key,
                          cache_folder, n_jobs, block_shape,
                          executable, ws_key,
                          use_bsub=use_bsub, eta=eta_[0])

    make_batch_jobs_step1(path, aff_key, out_key, mask_key,
                          cache_folder, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta_[1])

    make_batch_jobs_step2(cache_folder, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta_[2])

    make_batch_jobs_step3(path, out_key, cache_folder, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta_[3])

    make_batch_jobs_step4(cache_folder, n_jobs, executable, n_threads_ufd,
                          use_bsub=use_bsub, eta=eta_[4])

    make_batch_jobs_step5(path, out_key, cache_folder, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta_[5])

    if have_ws_job:
        make_batch_jobs_step6(path, aff_key, out_key, mask_key, ws_key,
                              cache_folder, n_jobs, executable,
                              use_bsub=use_bsub, eta=eta_[6])

        # make_batch_jobs_step7(cache_folder, n_jobs, executable,
        #                       use_bsub=use_bsub, eta=eta_[7])

        # make_batch_jobs_step8(path, ws_key, cache_folder, n_jobs, executable,
        #                       use_bsub=use_bsub, eta=eta_[8])

    make_master_job(n_jobs, executable, 'master.sh', have_ws_job)
