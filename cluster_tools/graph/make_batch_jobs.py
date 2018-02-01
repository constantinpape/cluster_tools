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


def make_batch_jobs_step1(labels_path, labels_key, graph_path,
                          tmp_folder, block_shape, n_jobs, n_scales, executable,
                          script_file='jobs_step1.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    make_executable('0_prepare.py')

    copy(os.path.join(file_dir, 'implementation/1_initial_graphs.py'), cwd)
    replace_shebang('1_initial_graphs.py', shebang)
    make_executable('1_initial_graphs.py')

    # def prepare(labels_path, labels_key, graph_path, n_jobs, tmp_folder, block_shape):
    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        f.write('./0_prepare.py %s %s %s --tmp_folder %s --block_shape %s --n_jobs %s --n_scales %s\n' %
                (labels_path, labels_key, graph_path, tmp_folder,
                 ' '.join(map(str, block_shape)), str(n_jobs), str(n_scales)))
        # TODO we need to check for success here !

        for job_id in range(n_jobs):
            command = './1_initial_graphs.py %s %s %s --block_file %s --block_shape %s' % \
                      (labels_path, labels_key, graph_path,
                       os.path.join(tmp_folder, '1_input_%i.npy' % job_id),
                       ' '.join(map(str, block_shape)))
            if use_bsub:
                log_file = 'logs/log_graph_step1_%i.log' % job_id
                err_file = 'error_logs/err_graph_step1_%i.err' % job_id
                f.write('bsub -J graph_step1_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


# generate jobs for all necessary scale levels
def make_batch_jobs_step2(graph_path, tmp_folder, block_shape, n_jobs, executable, n_scales,
                          script_file='jobs_step2.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/2_merge_graph_scales.py'), cwd)
    replace_shebang('2_merge_graph_scales.py', shebang)
    make_executable('2_merge_graph_scales.py')

    def make_jobs_scale(scale, f):
        for job_id in range(n_jobs):
            command = './2_merge_graph_scales.py %s %s --block_file %s --initial_block_shape %s' % \
                      (graph_path, str(scale),
                       os.path.join(tmp_folder, '2_input_s%i_%i.npy' % (job_id, scale)),
                       ' '.join(map(str, block_shape)))
            if use_bsub:
                log_file = 'logs/log_graph_step2_scale%i_%i.log' % (job_id, scale)
                err_file = 'error_logs/err_graph_step2_scale%i_%i.err' % (job_id, scale)
                f.write('bsub -J graph_step2_scale%i_%i -We %i -o %s -e %s \'%s\' \n' %
                        (scale, job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    # we have 1-based indexing for scales!
    for scale in range(2, n_scales + 1):
        scale_file = script_file[:-3] + 'scale%i.sh' % scale
        with open(scale_file, 'w') as f:
            make_jobs_scale(scale, f)
        make_executable(scale_file)


def make_batch_jobs_step3(graph_path, n_scales, block_shape, n_threads, executable,
                          script_file='jobs_step3.sh', use_bsub=True, eta=5):
    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/3_merge_graph.py'), cwd)
    replace_shebang('3_merge_graph.py', shebang)
    make_executable('3_merge_graph.py')

    # def prepare(labels_path, labels_key, graph_path, n_jobs, tmp_folder, block_shape):
    with open(script_file, 'w') as f:
        command = './3_merge_graph.py %s %s --initial_block_shape %s --n_threads %s' % \
                  (graph_path, str(n_scales),
                   ' '.join(map(str, block_shape)), str(n_threads))
        if use_bsub:
            log_file = 'logs/log_graph_step3.log'
            err_file = 'error_logs/err_graph_step3.err'
            f.write('bsub -n %i -J graph_step3 -We %i -o %s -e %s \'%s\' \n' %
                    (n_threads, eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


def make_batch_jobs_step4(graph_path, n_scales, n_threads, block_shape, executable,
                          script_file='jobs_step4.sh', use_bsub=True, eta=5):
    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/4_map_edge_ids.py'), cwd)
    replace_shebang('4_map_edge_ids.py', shebang)
    make_executable('4_map_edge_ids.py')

    # def prepare(labels_path, labels_key, graph_path, n_jobs, tmp_folder, block_shape):
    with open(script_file, 'w') as f:
        for scale in range(1, n_scales + 1):
            command = './4_map_edge_ids.py %s %s --initial_block_shape %s --n_threads %s' % \
                      (graph_path, str(scale),
                       ' '.join(map(str, block_shape)), str(n_threads))
            if use_bsub:
                log_file = 'logs/log_graph_step4_scale%i.log' % scale
                err_file = 'error_logs/err_graph_step4_scale%i.err' % scale
                f.write('bsub -n %i -J graph_step4_scale%i -We %i -o %s -e %s \'%s\' \n' %
                        (n_threads, scale, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


def make_master_job(n_jobs, n_scales, executable, script_file):
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
        f.write('./master_job.py %i %i\n' % (n_jobs, n_scales))
    make_executable(script_file)


def make_batch_jobs(labels_path, labels_key, graph_path, tmp_folder,
                    block_shape, n_scales, n_jobs, executable,
                    n_threads_merge=1, eta=5, use_bsub=True):

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

    make_batch_jobs_step1(labels_path, labels_key, graph_path,
                          tmp_folder, block_shape, n_jobs,
                          n_scales, executable,
                          use_bsub=use_bsub, eta=eta_[0])

    # we only need additional jobs if we have more than one scale level
    if n_scales > 1:
        make_batch_jobs_step2(graph_path, tmp_folder, block_shape, n_jobs, executable,
                              n_scales, use_bsub=use_bsub, eta=eta_[1])

    make_batch_jobs_step3(graph_path, n_scales, block_shape, n_threads_merge, executable,
                          use_bsub=use_bsub, eta=eta_[2])

    make_batch_jobs_step4(graph_path, n_scales, n_threads_merge, block_shape, executable,
                          use_bsub=use_bsub, eta=eta_[3])

    make_master_job(n_jobs, n_scales, executable, 'master.sh')
