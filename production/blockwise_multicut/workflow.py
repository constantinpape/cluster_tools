import os
import luigi
from .graph import GraphWorkflow
from .features import FeaturesWorkflow


class BlockwiseMulticutWorkflow(luigi.Task):
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    ws_key = luigi.Parameter()
    out_key = luigi.Parameter()
    max_scale = luigi.IntParameter()
    max_jobs = luigi.Parameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        graph_path = os.path.join(self.tmp_folder, 'graph.n5')
        features_path = os.path.join(self.tmp_folder, 'features.n5')
        graph_task = GraphWorkflow(path=self.path,
                                   ws_key=self.ws_key,
                                   out_path=graph_path,
                                   max_scale=self.max_scale,
                                   max_jobs=self.max_jobs,
                                   config_path=self.config_path,
                                   tmp_folder=self.tmp_folder,
                                   dependency=self.dependency,
                                   time_estimate=self.time_estimate,
                                   run_local=self.requires)
        features_task = FeaturesWorkflow(path=self.path,
                                         aff_key=self.aff_key,
                                         ws_key=self.ws_key,
                                         graph_path=graph_path,
                                         out_path=features_path,
                                         max_scale=self.max_scale,
                                         max_jobs=self.max_jobs,
                                         config_path=self.config_path,
                                         tmp_folder=self.tmp_folder,
                                         dependency=graph_task,
                                         time_estimate=self.time_estimate,
                                         run_local=self.requires)
        return features_task

    # just write a dummy file
    def run(self):
        out_path = self.input().path
        assert os.path.exists(out_path)
        res_file = self.output().path
        with open(res_file, 'w') as f:
            f.write('Success')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'blockwise_multicut_workflow.log'))
