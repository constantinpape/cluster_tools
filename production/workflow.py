import json
import os
import luigi

from .components import ComponentsWorkflow
from .watershed import FillingWatershedTask
from .relabel import RelabelWorkflow
from .stitching import ConsensusStitchingWorkflow
from .evaluation import SkeletonEvaluationTask
from .util import make_dirs
# from .util import DummyTask


class Workflow(luigi.WrapperTask):

    # path to the n5 file and keys
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    mask_key = luigi.Parameter()
    ws_key = luigi.Parameter()
    seg_key = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    # path to the configuration
    # TODO allow individual paths for individual blocks
    config_path = luigi.Parameter()
    tmp_folder_ws = luigi.Parameter()
    tmp_folder_seg = luigi.Parameter()
    # for evaluation
    skeleton_keys = luigi.ListParameter(default=[])
    # FIXME default does not work; this still needs to be specified
    # TODO different time estimates for different sub-tasks
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        # make the tmp, log and err dicts if necessary
        make_dirs(self.tmp_folder_ws)
        make_dirs(self.tmp_folder_seg)

        components_task = ComponentsWorkflow(path=self.path, aff_key=self.aff_key,
                                             mask_key=self.mask_key, out_key=self.ws_key,
                                             max_jobs=self.max_jobs, config_path=self.config_path,
                                             tmp_folder=self.tmp_folder_ws,
                                             time_estimate=self.time_estimate,
                                             run_local=self.run_local)
        ws_task = FillingWatershedTask(path=self.path, aff_key=self.aff_key,
                                       seeds_key=self.ws_key, mask_key=self.mask_key,
                                       max_jobs=self.max_jobs, config_path=self.config_path,
                                       tmp_folder=self.tmp_folder_ws,
                                       dependency=components_task,
                                       time_estimate=self.time_estimate,
                                       run_local=self.run_local)
        relabel_task = RelabelWorkflow(path=self.path, key=self.ws_key,
                                       max_jobs=self.max_jobs, config_path=self.config_path,
                                       tmp_folder=self.tmp_folder_ws,
                                       dependency=ws_task,
                                       time_estimate=self.time_estimate,
                                       run_local=self.run_local)
        stitch_task = ConsensusStitchingWorkflow(path=self.path,
                                                 aff_key=self.aff_key, ws_key=self.ws_key,
                                                 out_key=self.seg_key, max_jobs=self.max_jobs,
                                                 config_path=self.config_path,
                                                 tmp_folder=self.tmp_folder_seg,
                                                 dependency=relabel_task,
                                                 # dependency=DummyTask(),
                                                 time_estimate=self.time_estimate,
                                                 run_local=self.run_local)
        #
        if self.skeleton_keys:
            with open(self.config_path) as f:
                n_threads = json.load(f)['n_threads']
            eval_task = SkeletonEvaluationTask(path=self.path,
                                               seg_key=self.seg_key,
                                               skeleton_keys=self.skeleton_keys,
                                               n_threads=n_threads,
                                               tmp_folder=self.tmp_folder_seg,
                                               dependency=stitch_task,
                                               time_estimate=self.time_estimate,
                                               run_local=self.run_local)
            return eval_task
        #
        else:
            return stitch_task


def write_default_config(path,
                         # parameters for block shapes / shifts and chunks
                         block_shape=[50, 512, 512],
                         chunks=[25, 256, 256],
                         block_shape2=[75, 768, 768],
                         block_shift=[37, 384, 385],
                         # parameters for affinities used for components
                         boundary_threshold=.05,
                         aff_slices=[[0, 12], [12, 13]],
                         invert_channels=[True, False],
                         # parameters for watershed
                         boundary_threshold2=.2,
                         sigma_maxima=2.6,
                         size_filter=25,
                         # parameters for consensus stitching
                         weight_merge_edges=False,
                         weight_mulitcut_edges=False,
                         weighting_exponent=1.,
                         merge_threshold=.8,
                         affinity_offsets=[[-1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]],
                         # parameter for lifted multicut in consensus stitching
                         # (by default rf os set to None, which means lmc is not used)
                         lifted_rf_path=None,
                         lifted_nh=2,
                         # general parameter
                         n_threads=16):
    """
    Write the minimal config for consensus stitching workflow
    """
    try:
        with open(path) as f:
            config = json.load(f)
    except Exception:
        config = {}

    if lifted_rf_path is not None:
        assert os.path.exists(lifted_rf_path), lifted_rf_path

    config.update({'block_shape': block_shape,
                   'block_shape2': block_shape2,
                   'block_shift': block_shift,
                   'chunks': chunks,
                   'boundary_threshold': boundary_threshold,
                   'aff_slices': aff_slices,
                   'invert_channels': invert_channels,
                   'boundary_threshold2': boundary_threshold2,
                   'sigma_maxima': sigma_maxima,
                   'size_filter': size_filter,
                   'weight_merge_edges': weight_merge_edges,
                   'weight_multicut_edges': weight_mulitcut_edges,
                   'weighting_exponent': weighting_exponent,
                   'merge_threshold': merge_threshold,
                   'affinity_offsets': affinity_offsets,
                   'lifted_rf_path': lifted_rf_path,
                   'lifted_nh': lifted_nh,
                   'n_threads': n_threads})
    with open(path, 'w') as f:
        json.dump(config, f)


def write_dt_components_config(path,
                               # parameters for affinities used for components
                               # (defaults are different than for the thresholding based variant)
                               boundary_threshold=.2,
                               aff_slices=[[0, 3], [12, 13]],
                               invert_channels=[True, False],
                               resolution=(40., 4., 4.),
                               distance_threshold=40,
                               sigma=0.):
    """
    Write config to run the dt components workflow.
    Assumes that a default config already exits (otherwise writes it)
    """
    try:
        with open(path) as f:
            config = json.load(f)
    except Exception:
        write_default_config(path)
        with open(path) as f:
            config = json.load(f)

    config.update({'boundary_threshold': boundary_threshold,
                   'aff_slices': aff_slices,
                   'invert_channels': invert_channels,
                   'resolution': resolution,
                   'distance_threshold': distance_threshold,
                   'sigma': sigma})
    with open(path, 'w') as f:
        json.dump(config, f)
