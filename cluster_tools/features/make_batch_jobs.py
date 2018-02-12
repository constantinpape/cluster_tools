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


def make_batch_jobs_step1(graph_path, graph_key, out_path, out_key,
                          data_path, data_key, labels_path, labels_key,
                          tmp_folder, block_shape, n_jobs1, n_jobs2, executable,
                          script_file='jobs_step1.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/0_prepare.py'), cwd)
    replace_shebang('0_prepare.py', shebang)
    make_executable('0_prepare.py')

    copy(os.path.join(file_dir, 'implementation/1_block_features.py'), cwd)
    replace_shebang('1_block_features.py', shebang)
    make_executable('1_block_features.py')

    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        f.write('./0_prepare.py %s %s %s %s --block_shape %s --n_jobs1 %s --n_jobs2 %s --tmp_folder %s \n' %
                (graph_path, graph_key, out_path, out_key,
                 ' '.join(map(str, block_shape)),
                 str(n_jobs1), str(n_jobs2), tmp_folder))
        # TODO we need to check for success here !

        subgraph_prefix = os.path.join(graph_path, "sub_graphs/s0/block_")
        for job_id in range(n_jobs1):
            command = './1_block_features.py %s %s %s %s %s --offset_file %s --block_file %s --out_path %s' % \
                      (subgraph_prefix, data_path, data_key, labels_path, labels_key,
                       os.path.join(tmp_folder, 'offsets.json'),
                       os.path.join(tmp_folder, '1_input_%i.npy' % job_id),
                       out_path)
            if use_bsub:
                log_file = 'logs/log_features_step1_%i.log' % job_id
                err_file = 'error_logs/err_features_step1_%i.err' % job_id
                f.write('bsub -J features_step1_%i -We %i -o %s -e %s \'%s\' \n' %
                        (job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')

    make_executable(script_file)


# generate jobs for all necessary scale levels
def make_batch_jobs_step2(graph_path, out_path, out_key, tmp_folder, n_jobs,
                          n_threads, executable,
                          script_file='jobs_step2.sh', use_bsub=True, eta=5):

    # copy the relevant files
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    assert os.path.exists(executable), "Could not find python at %s" % executable
    shebang = '#! %s' % executable

    copy(os.path.join(file_dir, 'implementation/2_merge_features.py'), cwd)
    replace_shebang('2_merge_features.py', shebang)
    make_executable('2_merge_features.py')

    graph_block_prefix = os.path.join(graph_path, 'sub_graphs', 's0', 'block_')
    features_tmp_prefix = os.path.join(out_path, 'blocks', 'block_')
    with open(script_file, 'w') as f:
        f.write('#! /bin/bash\n')
        for job_id in range(n_jobs):
            command = './2_merge_features.py %s %s %s %s --input_file %s --n_threads %s' % \
                      (graph_block_prefix, features_tmp_prefix, out_path, out_key,
                       os.path.join(tmp_folder, '2_input_%i.npy' % job_id),
                       str(n_threads))
            if use_bsub:
                log_file = 'logs/log_features_step2_%i.log' % job_id
                err_file = 'error_logs/err_features_step2_%i.err' % job_id
                f.write('bsub -n %i -J features_step2_%i -We %i -o %s -e %s \'%s\' \n' %
                        (n_threads, job_id, eta, log_file, err_file, command))
            else:
                f.write(command + '\n')
    make_executable(script_file)


def make_master_job(n_jobs1, n_jobs2, executable, script_file):
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
        f.write('./master_job.py %i %i\n' % (n_jobs1, n_jobs2))
    make_executable(script_file)


def make_batch_jobs(graph_path, graph_key, out_path, out_key,
                    data_path, data_key, labels_path, labels_key,
                    tmp_folder, block_shape, n_jobs1, n_jobs2,
                    n_threads2, executable,
                    n_threads_merge=1, eta=5, use_bsub=True):

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

    make_batch_jobs_step1(graph_path, graph_key, out_path, out_key,
                          data_path, data_key, labels_path, labels_key,
                          tmp_folder, block_shape, n_jobs1, n_jobs2, executable,
                          use_bsub=use_bsub, eta=eta_[0])

    make_batch_jobs_step2(graph_path, out_path, out_key, tmp_folder, n_jobs2,
                          n_threads2, executable,
                          use_bsub=use_bsub, eta=eta_[1])

    make_master_job(n_jobs1, n_jobs2, executable, 'master.sh')
