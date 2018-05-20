import os
import luigi

from .compute_merge_votes import MergeVotesTask
from .compute_merges import MergesTask
from ..write import WriteAssignmentTask


class ConsensusStitchingWorkflow(luigi.Task):

    # path to the n5 file and keys
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    ws_key = luigi.Parameter()
    out_key = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    # path to the configuration
    # TODO allow individual paths for individual blocks
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        merge_vote_task = MergeVotesTask(path=self.path,
                                         aff_key=self.aff_key,
                                         ws_key=self.ws_key,
                                         out_key=self.out_key,
                                         max_jobs=self.max_jobs,
                                         config_path=self.config_path,
                                         tmp_folder=self.tmp_folder,
                                         dependency=self.dependency,
                                         time_estimate=self.time_estimate,
                                         run_local=self.run_local)
        merge_task = MergesTask(path=self.path,
                                out_key=self.out_key,
                                max_jobs=self.max_jobs,
                                config_path=self.config_path,
                                tmp_folder=self.tmp_folder,
                                dependency=merge_vote_task,
                                time_estimate=self.time_estimate,
                                run_local=self.run_local)
        write_task = WriteAssignmentTask(path=self.path,
                                         in_key=self.ws_key,
                                         out_key=self.out_key,
                                         config_path=self.config_path,
                                         max_jobs=self.max_jobs,
                                         tmp_folder=self.tmp_folder,
                                         identifier='write_consensus_stitching',
                                         dependency=merge_task,
                                         time_estimate=self.time_estimate,
                                         run_local=self.run_local)
        return write_task

    # just write a dummy file
    def run(self):
        out_path = self.input().path
        assert os.path.exists(out_path)
        res_file = self.output().path
        with open(res_file, 'w') as f:
            f.write('Success')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'consensus_stitching_workflow.log'))
