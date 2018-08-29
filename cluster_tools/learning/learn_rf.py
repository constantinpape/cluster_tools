#! /bin/python

import os
import sys
import json
import pickle

import numpy as np
import luigi
from sklearn.ensemble import RandomForestClassifier

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Learning Tasks
#

# TODO implement graph extraction with ignore label 0
class LearnRFBase(luigi.Task):
    """ LearnRF base class
    """

    task_name = 'learn_rf'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    features_dict = luigi.DictParameter()
    labels_dict = luigi.DictParameter()
    output_path = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'n_trees': 100})
        return config

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        assert self.features_dict.keys() == self.labels_dict.keys()

        # NOTE we have to turn the luigi dict parameters into normal python dicts
        # in order to json serialize them
        config.update({'features_dict': {key: val for key, val in self.features_dict.items()},
                       'labels_dict': {key: val for key, val in self.labels_dict.items()},
                       'output_path': self.output_path})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class LearnRFLocal(LearnRFBase, LocalTask):
    """ LearnRF on local machine
    """
    pass


class LearnRFSlurm(LearnRFBase, SlurmTask):
    """ LearnRF on slurm cluster
    """
    pass


class LearnRFLSF(LearnRFBase, LSFTask):
    """ LearnRF on lsf cluster
    """
    pass


#
# Implementation
#


def learn_rf(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    features_dict = config['features_dict']
    labels_dict = config['labels_dict']
    output_path = config['output_path']
    n_threads = config['threads_per_job']
    n_trees = config.get('n_trees', 100)

    features = []
    labels = []

    # TODO enable multiple feature paths
    # NOTE we assert that keys of boyh dicts are identical in the main class
    for key, feat_path in features_dict.items():
        label_path = labels_dict[key]
        fu.log("reading featurs from %s:%s, labels from %s:%s" % tuple(feat_path + label_path))

        with vu.file_reader(feat_path[0]) as f:
            ds = f[feat_path[1]]
            ds.n_threads = n_threads
            feats = ds[:]

        with vu.file_reader(label_path[0]) as f:
            ds = f[label_path[1]]
            ds.n_threads = n_threads
            label = ds[:]
        assert len(label) == len(feats)

        # check if we have an ignore label
        ignore_mask = label != -1
        n_ignore = np.sum(ignore_mask)
        if n_ignore < ignore_mask.size:
            fu.log("removing %i examples due to ignore mask" % n_ignore)
            feats = feats[ignore_mask]
            label = label[ignore_mask]

        features.append(feats)
        labels.append(label)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    fu.log("start learning random forest with %i examples and %i features" % features.shape)
    rf = RandomForestClassifier(n_estimators=n_trees,
                                n_jobs=n_threads)
    rf.fit(features, labels)

    fu.log("saving random forest to %s" % output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(rf, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    learn_rf(job_id, path)
