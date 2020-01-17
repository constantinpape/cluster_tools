import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import check_sub_graphs as check_tasks


# TODO fail if check does not pass
class CheckSubGraphsWorkflow(WorkflowBase):
    """ Check that watershed only has single connected
    component per label. Currently, does NOT work for
    two pass watershed.
    """
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    subgraph_key = luigi.Parameter()

    def requires(self):
        check_task = getattr(check_tasks, self._get_task_name('CheckSubGraphs'))
        dep = check_task(ws_path=self.ws_path, ws_key=self.ws_key,
                         graph_path=self.graph_path, subgraph_key=self.subgraph_key,
                         dependency=self.dependency,
                         tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir)
        return dep

    def run(self):
        violating_blocks = []
        for job_id in range(self.max_jobs):
            path = os.path.join(self.tmp_folder, 'failed_blocks_job_%i.json' % job_id)
            with open(path, 'r') as f:
                violating_blocks.extend(json.load(f))

        with open(self.output().path, 'w') as f:
            if violating_blocks:
                f.write('found %i violating blocks: \n' % len(violating_blocks))
                f.write('%s \n' % str(violating_blocks))
                raise RuntimeError("Sub-graph extraction failed!")
            else:
                f.write('found no violating blocks: \n')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'check_sub_graphs_workflow.log'))

    @staticmethod
    def get_config():
        configs = super(CheckSubGraphsWorkflow, CheckSubGraphsWorkflow).get_config()
        configs.update({'check_sub_graphs': check_tasks.CheckSubGraphsLocal.default_task_config()})
        return configs
