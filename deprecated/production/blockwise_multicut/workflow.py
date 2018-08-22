import os
import json
import luigi
from .graph import GraphWorkflow
from .features import FeaturesWorkflow
from .costs import CostsTask
from .multicut import MulticutWorkflow


class BlockwiseMulticutWorkflow(luigi.Task):
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    ws_key = luigi.Parameter()
    out_key = luigi.Parameter()
    max_jobs = luigi.Parameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):

        # hardcoded paths for graph, features and costs
        graph_path = os.path.join(self.tmp_folder, 'graph.n5')
        features_path = os.path.join(self.tmp_folder, 'features.n5')
        costs_path = os.path.join(self.tmp_folder, 'costs.n5')

        # get max scale and nax jobs mc from config
        with open(self.config_path) as f:
            config = json.load(f)
            max_jobs_mc = config.get('max_jobs_mc', 4)
            max_scale = config.get('max_scale', 0)

        graph_task = GraphWorkflow(path=self.path,
                                   ws_key=self.ws_key,
                                   out_path=graph_path,
                                   max_scale=max_scale,
                                   max_jobs=self.max_jobs,
                                   config_path=self.config_path,
                                   tmp_folder=self.tmp_folder,
                                   dependency=self.dependency,
                                   time_estimate=self.time_estimate,
                                   run_local=self.run_local)
        features_task = FeaturesWorkflow(path=self.path,
                                         aff_key=self.aff_key,
                                         ws_key=self.ws_key,
                                         graph_path=graph_path,
                                         out_path=features_path,
                                         max_scale=max_scale,
                                         max_jobs=self.max_jobs,
                                         config_path=self.config_path,
                                         tmp_folder=self.tmp_folder,
                                         dependency=graph_task,
                                         time_estimate=self.time_estimate,
                                         run_local=self.run_local)
        costs_task = CostsTask(features_path=features_path,
                               graph_path=graph_path,
                               out_path=costs_path,
                               config_path=self.config_path,
                               tmp_folder=self.tmp_folder,
                               dependency=features_task,
                               time_estimate=self.time_estimate,
                               run_local=self.run_local)
        # multicut needs to be run with less jobs
        mc_task = MulticutWorkflow(path=self.path,
                                   out_key=self.out_key,
                                   graph_path=graph_path,
                                   costs_path=costs_path,
                                   max_scale=max_scale,
                                   max_jobs=max_jobs_mc,
                                   config_path=self.config_path,
                                   tmp_folder=self.tmp_folder,
                                   dependency=costs_task,
                                   time_estimate=self.time_estimate,
                                   run_local=self.run_local)
        return mc_task

    # we check for the existence of the node labeling
    def output(self):
        return luigi.LocalTarget(os.path.join(self.path, self.out_key))
