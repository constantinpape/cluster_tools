import luigi
import os

from .solve_subproblems import SolveSubproblemTask
from .reduce_problem import ReduceProblemTask
from .global_problem import GlobalProblemTask


class MulticutWorkflow(luigi.Task):

    path = luigi.Parameter()
    out_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    costs_path = luigi.Parameter()
    max_scale = luigi.IntParameter()
    max_jobs = luigi.IntParameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):

        reduce_tasks = []
        for scale in range(self.max_scale + 1):
            costs_path = self.costs_path if scale == 0 else \
                os.path.join(self.tmp_folder, 'merged_graph.n5', 's%i' % scale)  # TODO
            dependency = self.dependency if scale == 0 else \
                reduce_tasks[-1]
            sub_task = SolveSubproblemTask(graph_path=self.graph_path,
                                           costs_path=costs_path,
                                           scale=scale,
                                           max_jobs=self.max_jobs,
                                           config_path=self.config_path,
                                           tmp_folder=self.tmp_folder,
                                           dependency=dependency,
                                           time_estimate=self.time_estimate,
                                           run_local=self.run_local)
            reduce_task = ReduceProblemTask(graph_path=self.graph_path,
                                            costs_path=costs_path,
                                            scale=scale,
                                            config_path=self.config_path,
                                            tmp_folder=self.tmp_folder,
                                            dependency=sub_task,
                                            time_estimate=self.time_estimate,
                                            run_local=self.run_local)
            reduce_tasks.append(reduce_task)

        global_task = GlobalProblemTask(path=self.path,
                                        out_key=self.out_key,
                                        max_scale=self.max_scale,
                                        config_path=self.config_path,
                                        tmp_folder=self.tmp_folder,
                                        dependency=reduce_tasks[-1],
                                        time_estimate=self.time_estimate,
                                        run_local=self.run_local)

        return global_task

    # just write a dummy file
    def run(self):
        out_path = self.input().path
        assert os.path.exists(out_path)
        res_file = self.output().path
        with open(res_file, 'w') as f:
            f.write('Success')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'multicut_workflow.log'))
