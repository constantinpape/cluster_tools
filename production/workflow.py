import luigi

from .components import ComponentsWorkflow
from .watershed import FillingWatershedTask
from . util import make_dirs


class Workflow(luigi.WrapperTask):

    # path to the n5 file and keys
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    mask_key = luigi.Parameter()
    out_key = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    # path to the configuration
    # TODO allow individual paths for individual blocks
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    # FIXME default does not work; this still needs to be specified
    # TODO different time estimates for different sub-tasks
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        # make the tmp, log and err dicts if necessary
        make_dirs(self.tmp_folder)

        components_task = ComponentsWorkflow(path=self.path, aff_key=self.aff_key,
                                             mask_key=self.mask_key, out_key=self.out_key,
                                             max_jobs=self.max_jobs, config_path=self.config_path,
                                             tmp_folder=self.tmp_folder, time_estimate=self.time_estimate,
                                             run_local=self.run_local)
        ws_task = FillingWatershedTask(path=self.path, aff_key=self.aff_key,
                                       seeds_key=self.out_key, mask_key=self.mask_key,
                                       max_jobs=self.max_jobs, config_path=self.config_path,
                                       tmp_folder=self.tmp_folder, dependency=components_task,
                                       time_estimate=self.time_estimate,
                                       run_local=self.run_local)
        return ws_task
        # return components_task


# TODO helper function for config
