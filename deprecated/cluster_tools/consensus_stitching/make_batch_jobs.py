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


def make_batch_jobs_step0(path, ws_key, out_key, cache_folder,
                          n_jobs, block_shape, block_shift, chunks,
                          executable,
                          script_file='jobs_step0.sh',
                          use_bsub=True, eta=5):
    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    make_executable('0_prepare.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './0_prepare.py %s %s %s %s %i --block_shape %s --block_shift %s --chunks %s\n' % \
            (path, ws_key, out_key,
             cache_folder, n_jobs,
             ' '.join(map(str, block_shape)),
             ' '.join(map(str, block_shift)),
             ' '.join(map(str, chunks)))
        if use_bsub:
            log_file = 'logs/log_consensus_stitching_step0.log'
            err_file = 'error_logs/err_consensus_stitching_step0.err'
            f.write('bsub -J consensus_stitching_step0 -We %i -o %s -e %s \'%s\' \n' %
                    (eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step1(path, ws_key, aff_key,
                          cache_folder, n_jobs, executable,
                          script_file='jobs_step1.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/1_process_blocks.py'), cwd)
    replace_shebang('1_process_blocks.py', shebang)
    make_executable('1_process_blocks.py')

    def job_for_prefix(f, prefix):
        f.write('#! /bin/bash\n')

        for job_id in range(n_jobs):
            command = './1_process_blocks.py %s %s %s %s %i %s' % (path, ws_key, aff_key,
                                                                   cache_folder, job_id, prefix)
            if use_bsub:
                log_file = 'logs/log_consensus_stitching_step1_%s_%i.log' % (prefix, job_id)
                err_file = 'error_logs/err_consensus_stitching_step1_%s_%i.err' % (prefix, job_id)
                f.write('bsub -J consensus_stitching_step1_%s_%i -We %i -o %s -e %s \'%s\' \n' %
                        (prefix, job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    script_file1 = script_file.split('.')[0] + '_blocking1.sh'
    with open(script_file1, 'w') as f:
        job_for_prefix(f, '1_blocking1')
    make_executable(script_file1)

    script_file2 = script_file.split('.')[0] + '_blocking2.sh'
    with open(script_file2, 'w') as f:
        job_for_prefix(f, '1_blocking2')
    make_executable(script_file2)


def make_batch_jobs_step2(path, out_key, aff_key, cache_folder, n_jobs,
                          merge_threshold, n_threads, executable,
                          script_file='jobs_step2.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/2_compute_merges.py'), cwd)
    replace_shebang('2_compute_merges.py', shebang)
    make_executable('2_compute_merges.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')

        # TODO don't hardcode prefixes
        prefixes = '1_blocking1 1_blocking2'
        command = './2_compute_merges.py %s %s %s %i %f %i %s' % (path, out_key, cache_folder, n_jobs,
                                                                  merge_threshold, n_threads, prefixes)
        if use_bsub:
            log_file = 'logs/log_consensus_stitching_step2.log'
            err_file = 'error_logs/err_consensus_stitching_step2.err'
            f.write('bsub -n %i -J consensus_stitching_step2 -We %i -o %s -e %s \'%s\' \n' %
                    (n_threads, eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step3(path, ws_key, out_key, cache_folder, n_jobs, executable,
                          script_file='jobs_step3.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/3_write_segmentation.py'), cwd)
    replace_shebang('3_write_segmentation.py', shebang)
    make_executable('3_write_segmentation.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')

        for job_id in range(n_jobs):
            command = './3_write_segmentation.py %s %s %s %s %i' % (path, ws_key, out_key,
                                                                    cache_folder, job_id)
            if use_bsub:
                log_file = 'logs/log_consensus_stitching_step3_%i.log' % job_id
                err_file = 'error_logs/err_consensus_stitching_step3_%i.err' % job_id
                f.write('bsub -J consensus_stitching_step3_%i -We %i -o %s -e %s \'%s\' \n' %
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

    # TODO don't hardcode prefixes
    prefixes = '1_blocking1 1_blocking2'
    with open(script_file, 'w') as f:
        f.write('./master_job.py %i %s\n' % (n_jobs, prefixes))
    make_executable(script_file)


def make_batch_jobs(path, ws_key, aff_key, out_key,
                    cache_folder, n_jobs,
                    block_shape, block_shift, chunks,
                    merge_threshold,
                    executable, eta=15, n_threads=4,
                    use_bsub=True):

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

    make_batch_jobs_step0(path, ws_key, out_key, cache_folder,
                          n_jobs, block_shape, block_shift, chunks,
                          executable,
                          use_bsub=use_bsub, eta=eta_[0])

    make_batch_jobs_step1(path, ws_key, aff_key,
                          cache_folder, n_jobs, executable,
                          use_bsub=use_bsub, eta=eta_[1])

    make_batch_jobs_step2(path, out_key, aff_key, cache_folder, n_jobs,
                          merge_threshold, n_threads, executable,
                          use_bsub=use_bsub, eta=eta_[2])

    make_batch_jobs_step3(path, ws_key, out_key, cache_folder, n_jobs,
                          executable, use_bsub=use_bsub, eta=eta_[3])

    make_master_job(n_jobs, executable, 'master.sh')
