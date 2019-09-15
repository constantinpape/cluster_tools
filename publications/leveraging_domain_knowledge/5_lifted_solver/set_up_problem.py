import os
from concurrent import futures

import numpy as np
import nifty
import nifty.distributed as ndist
import nifty.graph.opt.lifted_multicut as nlmc
import z5py


def probs_to_costs(costs, beta=.5):
    """ Transform probabilities to costs (in-place)
    """
    p_min = 0.001
    p_max = 1. - p_min
    costs = (p_max - p_min) * costs + p_min
    # probabilities to costs, second term is boundary bias
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)
    return costs


def set_up_problem():
    path = './exp_data/exp_data.n5'

    # load the graph and edge probs
    f = z5py.File(path)
    ds = f['features']
    probs = ds[:, 0]
    g = ndist.Graph(os.path.join(path, 's0/graph'))
    graph = nifty.graph.UndirectedGraph(g.numberOfNodes + 1)
    graph.insertEdges(g.uvIds())

    # add lifted edges up to nhood 3
    nhood = 2
    obj = nlmc.liftedMulticutObjective(graph)
    obj.insertLiftedEdgesBfs(nhood)
    lifted_uvs = obj.liftedUvIds()
    print("Number of lifted edges:")
    print(len(lifted_uvs))

    chunks = (int(1e6), 2)
    f.create_dataset('s0/lifted_nh', data=lifted_uvs, chunks=chunks, compression='gzip')

    # set the lifted costs according to the mean prob. of the shortest path
    print("Calculating costs ...")

    def find_costs_from_sp(lifted_id):
        print(lifted_id, "/", len(lifted_uvs))
        sp = nifty.graph.ShortestPathDijkstra(graph)
        u, v = lifted_uvs[lifted_id]
        edge_path = sp.runSingleSourceSingleTarget(probs, u, v, False)
        max_prob = np.max(probs[edge_path])
        return max_prob

    # p = find_costs_from_sp(0)
    # print(p)
    # return

    #
    n_threads = 8
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(find_costs_from_sp, i) for i in range(len(lifted_uvs))]
        costs = np.array([t.result() for t in tasks])
    assert len(costs) == len(lifted_uvs)
    costs = probs_to_costs(costs)
    chunks = (int(1e6),)
    f.create_dataset('s0/lifted_costs', data=costs, chunks=chunks, compression='gzip')


if __name__ == '__main__':
    set_up_problem()
