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


def make_batch_jobs_step1(features_path, features_key, out_path, out_key,
                          n_jobs, tmp_folder, random_forest_path, executable,
                          n_threads_rf=8, script_file='jobs_step1.sh',
                          use_bsub=True, eta=5):

    if random_forest_path != '':
        assert os.path.exists(random_forest_path)
        with_rf = True
    else:
        with_rf = False

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    make_executable('0_prepare.py')

    copy(os.path.join(file_dir, 'implementation/1_rf.py'), cwd)
    replace_shebang('1_rf.py', shebang)
    make_executable('1_rf.py')

    def write_rf_jobs(f):
        for job_id in range(n_jobs):
            command = './1_rf.py %s %s %s %s %s --input_file %s --n_threads_rf %s' % \
                      (features_path, features_key, out_path, out_key, random_forest_path,
                       os.path.join(tmp_folder, '1_input_%i.npy' % job_id),
                       str(n_threads_rf))
            if use_bsub:
                log_file = 'logs/log_costs_step1_%i.log' % job_id
                err_file = 'error_logs/err_costs_step1_%i.err' % job_id
                f.write('bsub -n %i -J costs_step1_%i -We %i -o %s -e %s \'%s\' \n' %
                        (n_threads_rf, job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        f.write('./0_prepare.py %s %s %s %s --n_jobs %s --tmp_folder %s --random_forest_path %s \n' %
                (features_path, features_key, out_path, out_key,
                 str(n_jobs), tmp_folder, random_forest_path))
        if with_rf:
            write_rf_jobs(f)

    make_executable(script_file)
    return with_rf


# generate jobs for all necessary scale levels
def make_batch_jobs_step2(features_path, features_key, graph_path, graph_key,
                          out_path, out_key, with_rf, executable,
                          script_file='jobs_step2.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/2_postprocess_costs.py'), cwd)
    replace_shebang('2_postprocess_costs.py', shebang)
    make_executable('2_postprocess_costs.py')

    #  if we predict the edge probabilities with a random forests, we
    # store the costs in the out file already
    if with_rf:
        input_path = out_path
        input_key = out_key
        invert_inputs = 0
    # otherwise, we need to read the edge probabilities from the features
    else:
        input_path = features_path
        input_key = features_key
        invert_inputs = 1

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        command = './2_postprocess_costs.py %s %s %s %s %s %s %s %s --invert_inputs %s' % \
                  (input_path, input_key, features_path, features_key,
                   graph_path, graph_key, out_path, out_key,
                   str(invert_inputs))
        if use_bsub:
            log_file = 'logs/log_costs_step2.log'
            err_file = 'error_logs/err_costs_step2.err'
            f.write('bsub -J costs_step2 -We %i -o %s -e %s \'%s\' \n' %
                    (eta, log_file, err_file, command))
        else:
            f.write(command + '\n')
    make_executable(script_file)


def make_master_job(n_jobs, with_rf, executable, script_file):
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
        f.write('./master_job.py %i %i\n' % (n_jobs, with_rf))
    make_executable(script_file)


def make_batch_jobs(features_path, features_key, graph_path, graph_key,
                    random_forest_path, out_path, out_key, n_jobs, tmp_folder,
                    n_threads_rf, executable,
                    eta=5, use_bsub=True):

    assert isinstance(eta, (int, list, tuple))
    if isinstance(eta, (list, tuple)):
        assert len(eta) == 2
        assert all(isinstance(ee, int) for ee in eta)
        eta_ = eta
    else:
        eta_ = (eta,) * 2

    # clean logs
    if os.path.exists('error_logs'):
        rmtree('error_logs')
    os.mkdir('error_logs')

    if os.path.exists('logs'):
        rmtree('logs')
    os.mkdir('logs')

    with_rf = make_batch_jobs_step1(features_path, features_key, out_path, out_key,
                                    n_jobs, tmp_folder, random_forest_path,
                                    executable, n_threads_rf=n_threads_rf,
                                    use_bsub=use_bsub, eta=eta_[0])

    make_batch_jobs_step2(features_path,  features_key, graph_path, graph_key,
                          out_path, out_key, with_rf, executable,
                          use_bsub=use_bsub, eta=eta_[1])

    make_master_job(n_jobs, with_rf, executable, 'master.sh')
