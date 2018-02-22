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


# this only calls 0_prepare.py, slight misnomer....
def make_batch_jobs_step1(graph_path, graph_key, costs_path, costs_key,
                          block_shape, n_scales, tmp_folder, n_jobs, n_threads,
                          use_mc_costs, executable,
                          script_file='jobs_step1.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    make_executable('0_prepare.py')

    # FIXME for some reason, the log of this file is not properly read...
    use_bsub = False

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './0_prepare.py %s %s %s %s --initial_block_shape %s --n_scales %s --tmp_folder %s --n_jobs %s --n_threads %s --use_mc_costs %s\n' %\
                  (graph_path, graph_key, costs_path, costs_key,
                   ' '.join(map(str, block_shape)),
                   str(n_scales), tmp_folder, str(n_jobs), str(n_threads),
                   str(use_mc_costs))

        if use_bsub:
            log_file = 'logs/log_multicut_step1.log'
            err_file = 'error_logs/err_multicut_step1.err'
            # For now we don't gain anything by multi-threading
            # f.write('bsub -n %i -J multicut_step1 -We %i -o %s -e %s \'%s\' \n' %
            #         (n_threads, eta, log_file, err_file, command))
            f.write('bsub -J multicut_step1 -We %i -o %s -e %s \'%s\' \n' %
                    (eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    make_executable(script_file)


# generate jobs for all necessary scale levels
def make_batch_jobs_step2(graph_path, tmp_folder, n_scales,
                          agglomerator_key, n_threads,
                          n_jobs, block_shape, executable,
                          script_file='jobs_step2.sh',
                          use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/1a_solve_subproblems.py'), cwd)
    replace_shebang('1a_solve_subproblems.py', shebang)
    make_executable('1a_solve_subproblems.py')

    copy(os.path.join(file_dir, 'implementation/1b_reduce_problem.py'), cwd)
    replace_shebang('1b_reduce_problem.py', shebang)
    make_executable('1b_reduce_problem.py')

    def make_jobs_subproblem(scale, f):
        f.write('#! /bin/bash\n')
        if scale == 0:
            block_prefix = os.path.join(graph_path, 'sub_graphs', 's%i' % scale, 'block_')
        else:
            block_prefix = os.path.join(graph_path, 'merged_graphs', 's%i' % scale, 'block_')

        for job_id in range(n_jobs):
            command = './1a_solve_subproblems.py %s %s %s --tmp_folder %s --agglomerator_key %s --initial_block_shape %s --block_file %s' % \
                      (graph_path, block_prefix, str(scale),
                       tmp_folder,
                       '_'.join(agglomerator_key),
                       ' '.join(map(str, block_shape)),
                       os.path.join(tmp_folder, '2_input_s%i_%i.npy' % (scale, job_id)))
            if use_bsub:
                log_file = 'logs/log_multicut_step2_scale%i_%i.log' % (scale, job_id)
                err_file = 'error_logs/err_multicut_step2_scale%i_%i.err' % (scale, job_id)
                f.write('bsub -J multicut_step2_scale%i_%i -We %i -o %s -e %s \'%s\' \n' %
                        (scale, job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    def make_jobs_reduce(scale, f):
        f.write('#! /bin/bash\n')
        command = './1b_reduce_problem.py %s %s --tmp_folder %s --n_jobs %s --initial_block_shape %s --n_threads %s --cost_accumulation %s' % \
                  (graph_path, str(scale),
                   tmp_folder, str(n_jobs),
                   ' '.join(map(str, block_shape)),
                   str(n_threads),
                   'sum' if agglomerator_key[0] == 'multicut' else 'mean')
        if use_bsub:
            log_file = 'logs/log_multicut_step2_scale%i.log' % scale
            err_file = 'error_logs/err_multicut_step2_scale%i.err' % scale
            f.write('bsub -n %i -J multicut_step2_scale%i -We %i -o %s -e %s \'%s\' \n' %
                    (n_threads, scale, eta, log_file, err_file, command))
        else:
            f.write(command + '\n')

    for scale in range(n_scales):
        subproblem_file = script_file[:-3] + 'subproblem_scale%i.sh' % scale
        with open(subproblem_file, 'w') as f:
            make_jobs_subproblem(scale, f)
        make_executable(subproblem_file)

        reduce_file = script_file[:-3] + 'reduce_scale%i.sh' % scale
        with open(reduce_file, 'w') as f:
            make_jobs_reduce(scale, f)
        make_executable(reduce_file)


def make_batch_jobs_step3(out_path, node_labeling_key,
                          n_scales, tmp_folder,
                          n_threads, agglomerator_key, executable,
                          script_file='jobs_step3.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/2_solve_global.py'), cwd)
    replace_shebang('2_solve_global.py', shebang)
    make_executable('2_solve_global.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './2_solve_global.py %s %s %s --tmp_folder %s --agglomerator_key %s\n' %\
                  (out_path, node_labeling_key,
                   str(n_scales),
                   tmp_folder, '_'.join(agglomerator_key))

        if use_bsub:
            log_file = 'logs/log_multicut_step3.log'
            err_file = 'error_logs/err_multicut_step3.err'
            f.write('bsub -n %i -J multicut_step3 -We %i -o %s -e %s \'%s\' \n' %
                    (n_threads, eta, log_file, err_file, command))
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


def make_batch_jobs(graph_path, graph_key, costs_path, costs_key,
                    out_path, node_out_key,
                    block_shape, n_scales, tmp_folder, n_jobs, executable,
                    agglomerator_key=('multicut', 'kl'),
                    n_threads=1, eta=5, use_bsub=True):

    assert isinstance(agglomerator_key, tuple), agglomerator_key

    n_steps = 3
    assert isinstance(eta, (int, list, tuple))
    if isinstance(eta, (list, tuple)):
        assert len(eta) == n_steps
        assert all(isinstance(ee, int) for ee in eta)
        eta_ = eta
    else:
        eta_ = (eta,) * n_steps

    # clean logs
    if os.path.exists('error_logs'):
        rmtree('error_logs')
    os.mkdir('error_logs')

    if os.path.exists('logs'):
        rmtree('logs')
    os.mkdir('logs')

    make_batch_jobs_step1(graph_path, graph_key, costs_path, costs_key,
                          block_shape, n_scales, tmp_folder, n_jobs, n_threads,
                          use_mc_costs=1 if agglomerator_key[0] == 'multicut' else 0,
                          executable=executable,
                          use_bsub=use_bsub, eta=eta_[0])
    make_batch_jobs_step2(graph_path, tmp_folder, n_scales,
                          agglomerator_key, n_threads,
                          n_jobs, block_shape, executable,
                          use_bsub=use_bsub, eta=eta_[1])
    make_batch_jobs_step3(out_path, node_out_key,
                          n_scales, tmp_folder,
                          n_threads, agglomerator_key, executable,
                          use_bsub=use_bsub, eta=eta_[2])

    make_master_job(n_jobs, n_scales, executable, 'master.sh')
