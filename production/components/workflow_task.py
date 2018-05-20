import os
import luigi

from .compute_block_offsets import OffsetTask
from .threshold_components import ThresholdTask
from .merge_blocks import MergeTask
from .node_assignment import NodeAssignmentTask
from ..write import WriteAssignmentTask


class ComponentsWorkflow(luigi.Task):

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

        thresh_task = ThresholdTask(path=self.path, aff_key=self.aff_key,
                                    mask_key=self.mask_key, out_key=self.out_key,
                                    max_jobs=self.max_jobs, config_path=self.config_path,
                                    tmp_folder=self.tmp_folder, time_estimate=self.time_estimate,
                                    run_local=self.run_local)
        offset_task = OffsetTask(tmp_folder=self.tmp_folder, dependency=thresh_task,
                                 time_estimate=self.time_estimate, run_local=self.run_local)
        merge_task = MergeTask(path=self.path, out_key=self.out_key, config_path=self.config_path,
                               max_jobs=self.max_jobs, tmp_folder=self.tmp_folder,
                               dependency=offset_task,
                               time_estimate=self.time_estimate, run_local=self.run_local)
        assignment_task = NodeAssignmentTask(path=self.path, out_key=self.out_key, config_path=self.config_path,
                                             max_jobs=self.max_jobs, tmp_folder=self.tmp_folder,
                                             dependency=merge_task,
                                             time_estimate=self.time_estimate, run_local=self.run_local)
        write_task = WriteAssignmentTask(path=self.path, in_key=self.out_key,
                                         out_key=self.out_key, config_path=self.config_path,
                                         max_jobs=self.max_jobs, tmp_folder=self.tmp_folder,
                                         dependency=assignment_task,
                                         offset_path=os.path.join(self.tmp_folder, 'block_offsets.json'),
                                         time_estimate=self.time_estimate, run_local=self.run_local)
        return write_task

    # just write a dummy file
    def run(self):
        out_path = self.input().path
        assert os.path.exists(out_path)
        res_file = self.output().path
        with open(res_file, 'w') as f:
            f.write('Success')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'components_workflow.log'))
