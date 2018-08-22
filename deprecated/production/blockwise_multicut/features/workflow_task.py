import luigi
import os

from .initial_features import InitialFeaturesTask
from .merge_features import MergeFeaturesTask


class FeaturesWorkflow(luigi.Task):

    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    ws_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    out_path = luigi.Parameter()
    max_scale = luigi.IntParameter()
    max_jobs = luigi.Parameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):

        initial_task = InitialFeaturesTask(path=self.path,
                                           aff_key=self.aff_key,
                                           ws_key=self.ws_key,
                                           graph_path=self.graph_path,
                                           out_path=self.out_path,
                                           max_jobs=self.max_jobs,
                                           config_path=self.config_path,
                                           tmp_folder=self.tmp_folder,
                                           dependency=self.dependency,
                                           time_estimate=self.time_estimate,
                                           run_local=self.run_local)
        merge_task = MergeFeaturesTask(graph_path=self.graph_path,
                                       out_path=self.out_path,
                                       config_path=self.config_path,
                                       tmp_folder=self.tmp_folder,
                                       dependency=initial_task,
                                       time_estimate=self.time_estimate,
                                       run_local=self.run_local)
        return merge_task

    # just write a dummy file
    def run(self):
        out_path = self.input().path
        assert os.path.exists(out_path)
        res_file = self.output().path
        with open(res_file, 'w') as f:
            f.write('Success')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'features_workflow.log'))
