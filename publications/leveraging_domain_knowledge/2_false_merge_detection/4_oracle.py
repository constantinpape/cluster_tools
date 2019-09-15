import os
import luigi
import z5py

from cluster_tools.node_labels import NodeLabelWorkflow
from resolving import edges_from_skeletons, edges_to_problem
from resolving import combine_edges_and_costs


def precompute_skeleton_problems(scale=2):

    path = '/g/kreshuk/data/FIB25/data.n5'
    ws_key = 'volumes/segmentation/watershed'
    gt_key = 'volumes/segmentation/groundtruth_filtered'
    ass_key = 'node_labels/multitcut_filtered'

    labels_key = 'overlaps/wsgt'
    task = NodeLabelWorkflow

    t = task(tmp_folder='tmp_labels', config_dir='./configs',
             max_jobs=48, target='local',
             ws_path=path, ws_key=ws_key,
             input_path=path, input_key=gt_key,
             output_path=path, output_key=labels_key)
    luigi.build([t], local_scheduler=True)

    ws_key_scale = 'volumes/paintera/watershed/data/s%i' % scale
    skel_key = 'skeletons/s%i' % scale

    graph_path = '/g/kreshuk/data/FIB25/exp_data/mc.n5'
    graph_key = 's0/graph'

    out_key = 'resolving/edges_and_indicators'
    n_threads = 48

    edges_from_skeletons(path, ws_key_scale, labels_key,
                         skel_key, ass_key, out_key,
                         graph_path, graph_key, n_threads)


def get_max_costs(factor=1):
    path = '/g/kreshuk/data/FIB25/exp_data/mc.n5'
    f = z5py.File(path)
    ds = f['s0/costs']
    ds.n_threads = 8
    costs = ds[:]

    max_abs_cost = max([abs(costs.min()),
                        abs(costs.max())])
    max_attractive = factor * max_abs_cost
    max_repulsive = -1 * factor * max_abs_cost
    return max_attractive, max_repulsive


def make_lifted_problem(path, name, out_path, n_threads):
    # we make softlinks to graph and local costs, etc.
    ref_path = '/g/kreshuk/data/FIB25/exp_data/mc.n5'
    attrs = z5py.File(ref_path).attrs

    f = z5py.File(out_path)
    for k, v in attrs.items():
        f.attrs[k] = v

    g = f.require_group('s0')
    to_link = ['costs', 'graph', 'sub_graphs']
    for t in to_link:
        src = os.path.join(ref_path, 's0', t)
        dst = os.path.join(g.path, t)
        if os.path.exists(dst):
            continue
        os.symlink(src, dst)

    in_group = 'resolving/oracle/%s' % name
    out_key_uvs = 's0/lifted_nh_resolving'
    out_key_costs = 's0/lifted_costs_resolving'
    combine_edges_and_costs(path, in_group, out_path,
                            out_key_uvs, out_key_costs, n_threads)


# TODO we want multiple draws for the imperfect oracle
# in order to get some sense of the statistics
def oracle(recall=1., precision=1.):
    path = '/g/kreshuk/data/FIB25/data.n5'

    if recall == 1. and precision == 1.:
        name = 'perfect_oracle'
    else:
        name = 'oracle_rec%.02f_prec%.02f' % (recall, precision)

    key_in = 'resolving/edges_and_indicators'
    key_uv = 'resolving/oracle/%s/uvs' % name
    key_costs = 'resolving/oracle/%s/costs' % name

    max_attractive, max_repulsive = get_max_costs()
    print("Making costs with max attractive / repulsive values:")
    print(max_attractive, max_repulsive)

    n_threads = 48
    # extract subproblems
    edges_to_problem(path, path,
                     key_in, key_uv, key_costs,
                     precision, recall,
                     max_attractive, max_repulsive,
                     n_threads)

    # make full lifted problem
    out_path = os.path.join('/g/kreshuk/data/FIB25/exp_data',
                            '%s.n5' % name)
    make_lifted_problem(path, name, out_path, n_threads)


if __name__ == '__main__':
    precompute_skeleton_problems()
    oracle()
