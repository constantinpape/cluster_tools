import os
import datetime
import z5py
import nifty
import nifty.distributed as ndist
import nifty.graph.opt.lifted_multicut as nlmc


def compute_energy(name):
    exp_path = './exp_data/%s.n5' % name
    f = z5py.File(exp_path)

    g = ndist.Graph(os.path.join(exp_path, 's0', 'graph'))
    graph = nifty.graph.undirectedGraph(g.numberOfNodes + 1)
    graph.insertEdges(g.uvIds())

    costs = f['s0/costs'][:]
    lifted_costs = f['s0/lifted_costs_%s' % name][:]
    lifted_uvs = f['s0/lifted_nh_%s' % name][:]

    obj = nlmc.liftedMulticutObjective(graph)
    obj.setGraphEdgesCosts(costs)
    obj.setCosts(lifted_uvs, lifted_costs)

    path = '/g/kreshuk/data/FIB25/cutout.n5'
    res_key = 'node_labels/%s' % name
    with z5py.File(path) as f:
        node_labels = f[res_key][:]

    e = obj.evalNodeLabels(node_labels)
    return e


def compute_time(p):
    lines = []
    with open(p, 'r') as f:
        for l in f:
            lines.append(l)
    t0 = lines[0].split()[1][:-1]
    t1 = lines[-1].split()[1][:-1]

    t0 = datetime.datetime.strptime(t0, '%H:%M:%S.%f')
    t1 = datetime.datetime.strptime(t1, '%H:%M:%S.%f')
    t = t1 - t0
    return t


def eval_solver(name):
    p = './tmp_folders/%s/logs/solve_lifted_global_s0_0.log' % name
    t = compute_time(p)
    e = compute_energy(name)
    print("Solver:", name)
    print("Energy:", e)
    print("Time:", t)


def eval_hierarchical():
    name = 'hierarchical'
    e = compute_energy(name)
    print("Solver:", name)
    print("Energy:", e)

    p = './tmp_folders/hierarchical/logs/solve_lifted_subproblems_s0_0.log'
    t_sub = compute_time(p)
    print("T-Sub:", t_sub)

    p = './tmp_folders/hierarchical/logs/reduce_lifted_problem_s0_0.log'
    t_red = compute_time(p)
    print("T-Red:", t_red)

    p = './tmp_folders/hierarchical/logs/solve_lifted_global_s1_0.log'
    t_sol = compute_time(p)
    print("T-Sol:", t_sol)


if __name__ == '__main__':
    eval_solver('greedy-additive')
    print()
    eval_solver('kernighan-lin')
    print()
    eval_solver('fusion-moves')
    print()
    eval_hierarchical()
