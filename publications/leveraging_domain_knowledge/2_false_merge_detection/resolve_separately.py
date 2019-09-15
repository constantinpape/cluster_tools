import os
import luigi

from resolving import ResolvingWorkflow


def resolve_separately(identifier, max_jobs=48, target='local'):
    task = ResolvingWorkflow

    path = '/g/kreshuk/data/FIB25/data.n5'
    exp_path = '/g/kreshuk/data/FIB25/exp_data/mc.n5'

    # objects_group = 'resolving/oracle/perfect_oracle'
    objects_group = 'resolving/oracle/%s' % identifier

    assignment_in_key = 'node_labels/multitcut_filtered'
    assignment_out_key = 'node_labels/resolve_separately/%s' % identifier

    tmp_folder = './tmp_folders/tmp_resolve_separately_%s' % identifier
    os.makedirs(tmp_folder, exist_ok=True)

    # TODO write to actual output
    ws_key = 'volumes/segmentation/watershed'
    out_key = 'volumes/segmentation/resolve_separately/%s' % identifier

    t = task(tmp_folder=tmp_folder, config_dir='./configs',
             max_jobs=max_jobs, target=target,
             problem_path=exp_path, path=path,
             objects_group=objects_group,
             assignment_in_key=assignment_in_key,
             assignment_out_key=assignment_out_key,
             ws_key=ws_key, out_key=out_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Resolving failed"


if __name__ == '__main__':
    resolve_separately('perfect_oracle')
