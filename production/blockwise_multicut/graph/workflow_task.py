import luigi
import os

from .initial_subgraph import InitialSubgraphTask
from .merge_graph_scales import MergeSubgraphScalesTask
from .merge_graph import MergeGraphTask
from .map_edge_ids import MapEdgesTask


class GraphWorkflow(luigi.Task):

    path = luigi.Parameter()
    ws_key = luigi.Parameter()
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

        initial_task = InitialSubgraphTask(path=self.path, ws_key=self.ws_key,
                                           out_path=self.out_path,
                                           max_jobs=self.max_jobs, config_path=self.config_path,
                                           tmp_folder=self.tmp_folder, dependency=self.dependency,
                                           time_estimate=self.time_estimate,
                                           run_local=self.run_local)
        if self.max_scale > 0:
            scale_tasks = [initial_task]
            for scale in range(self.max_scale):
                scale_tasks.append(MergeSubgraphScalesTask(path=self.path, ws_key=self.ws_key,
                                                           out_path=self.out_path, scale=scale + 1,
                                                           max_jobs=self.max_jobs, config_path=self.config_path,
                                                           tmp_folder=self.tmp_folder, dependency=scale_tasks[-1],
                                                           time_estimate=self.time_estimate,
                                                           run_local=self.run_local))
        merge_task = MergeGraphTask(out_path=self.out_path, max_scale=self.max_scale,
                                    config_path=self.config_path, tmp_folder=self.tmp_folder,
                                    dependency=scale_tasks[-1],
                                    time_estimate=self.time_estimate, run_local=self.run_local)
        map_task = MapEdgesTask(out_path=self.out_path, max_scale=self.max_scale,
                                config_path=self.config_path, tmp_folder=self.tmp_folder,
                                dependency=merge_task,
                                time_estimate=self.time_estimate, run_local=self.run_local)
        return map_task

    # just write a dummy file
    def run(self):
        out_path = self.input().path
        assert os.path.exists(out_path)
        res_file = self.output().path
        with open(res_file, 'w') as f:
            f.write('Success')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'graph_workflow.log'))
