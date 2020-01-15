#! /bin/python

import os
import sys
import json
import pickle

# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
from elf.util import set_numpy_threads
set_numpy_threads(1)
import numpy as np

import luigi
import vigra
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# ObjectDistances Tasks
#

class ObjectDistancesBase(luigi.Task):
    """ ObjectDistances base class
    """

    task_name = 'object_distances'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    morphology_path = luigi.Parameter()
    morphology_key = luigi.Parameter()
    max_distance = luigi.FloatParameter()
    resolution = luigi.ListParameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'id_chunks': 2000})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape = self.global_config_values()[:2]
        self.init(shebang)

        # load the object_distances config
        config = self.get_task_config()
        # update the config with input paths and keys, etc.
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'morphology_path': self.morphology_path, 'morphology_key': self.morphology_key,
                       'max_distance': self.max_distance, 'resolution': self.resolution,
                       'tmp_folder': self.tmp_folder})

        with vu.file_reader(self.input_path, 'r') as f:
            n_labels = f[self.input_key].attrs['maxId'] + 1

        id_chunks = config['id_chunks']
        if self.n_retries == 0:
            block_list = vu.blocks_in_volume([n_labels], [id_chunks])
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        self._write_log('scheduling %i blocks to be processed' % len(block_list))
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class ObjectDistancesLocal(ObjectDistancesBase, LocalTask):
    """
    ObjectDistances on local machine
    """
    pass


class ObjectDistancesSlurm(ObjectDistancesBase, SlurmTask):
    """
    ObjectDistances on slurm cluster
    """
    pass


class ObjectDistancesLSF(ObjectDistancesBase, LSFTask):
    """
    ObjectDistances on lsf cluster
    """
    pass


#
# Implementation
#


def _labels_and_distances(ds, bb, resolution, label_id):
    labels = ds[bb].astype('uint32')
    object_mask = (labels == label_id).astype('uint32')
    distances = vigra.filters.distanceTransform(object_mask, pixel_pitch=resolution)
    return labels, distances


def _get_faces():
    faces = [np.s_[0, :, :], np.s_[-1, :, :],
             np.s_[:, 0, :], np.s_[:, -1, :],
             np.s_[:, :, 0], np.s_[:, :, -1]]
    return faces


def _compute_face_distances(distances):
    # I probably have implemented this somewheres else already ...
    face_distances = []
    faces = _get_faces()
    for face in faces:
        face_distances.append(distances[face].min())
    return face_distances


def _enlarge_bb(bb, face_distances, resolution, shape, max_distance):
    enlarged = []
    face_id = 0
    for dim, b in enumerate(bb):
        start, stop = b.start, b.stop
        res = resolution[dim]

        fdist = face_distances[face_id]
        if fdist < max_distance:
            start -= (max_distance - fdist) / res
            start = max(start, 0)
        face_id += 1

        fdist = face_distances[face_id]
        if fdist < max_distance:
            stop += (max_distance - fdist) / res
            stop = min(stop, shape[dim])
        face_id += 1

        enlarged.append(slice(int(start), int(stop)))
    return tuple(enlarged)


def _object_distances(label_id, ds, bb_start, bb_stop,
                      max_distance, resolution):
    bb = tuple(slice(sta, sto) for sta, sto in zip(bb_start[label_id], bb_stop[label_id]))
    labels, distances = _labels_and_distances(ds, bb, resolution, label_id)

    # compute all face distances and the
    face_distances = _compute_face_distances(distances)
    min_bd_distance = min(face_distances)

    # enlarge the bounding box if we don't have max distances to all side
    if min_bd_distance < max_distance:
        bb = _enlarge_bb(bb, face_distances, resolution, ds.shape, max_distance)
        labels, distances = _labels_and_distances(ds, bb, resolution, label_id)
        face_distances = _compute_face_distances(distances)

    object_ids = np.unique(labels)
    if 0 in object_ids:
        object_ids = object_ids[1:]
    object_distances = vigra.analysis.extractRegionFeatures(distances, labels,
                                                            features=['Minimum'])['Minimum']
    dist_dict = {(label_id, obj_id): object_distances[obj_id]
                 for obj_id in object_ids if label_id < obj_id}
    dist_dict = {k: v for k, v in dist_dict.items() if v < max_distance}
    return dist_dict


def _distances_id_chunks(blocking, block_id, ds_in,
                         bb_start, bb_stop, max_distance, resolution):
    block = blocking.getBlock(block_id)
    id_start, id_stop = block.begin[0], block.end[0]
    # skip 0, which is the ignore label
    id_start = max(id_start, 1)

    block_distances = {}
    for label_id in range(id_start, id_stop):
        dists = _object_distances(label_id, ds_in,
                                  bb_start, bb_stop,
                                  max_distance, resolution)
        block_distances.update(dists)

    return block_distances


def object_distances(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']

    morphology_path = config['morphology_path']
    morphology_key = config['morphology_key']

    max_distance = config['max_distance']
    resolution = config['resolution']

    block_list = config['block_list']
    id_chunks = config['id_chunks']
    tmp_folder = config['tmp_folder']

    with vu.file_reader(morphology_path, 'r') as f:
        morpho = f[morphology_key][:]
        bb_start = morpho[:, 5:8].astype('uint64')
        bb_stop = morpho[:, 8:11].astype('uint64') + 1

    with vu.file_reader(input_path, 'r') as f:

        ds_in = f[input_key]
        n_labels = ds_in.attrs['maxId'] + 1

        # get the blocking
        blocking = nt.blocking([0], [n_labels], [id_chunks])

        res_dict = {}
        for block_id in block_list:
            block_dict = _distances_id_chunks(blocking, block_id, ds_in,
                                              bb_start, bb_stop, max_distance, resolution)
            res_dict.update(block_dict)

        out_path = os.path.join(tmp_folder, 'object_distances_%i.pkl' % job_id)
        with open(out_path, 'wb') as f:
            pickle.dump(res_dict, f)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    object_distances(job_id, path)
