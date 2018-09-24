import nifty.graph.opt.multicut as nmc
import nifty.graph.opt.lifted_multicut as nlmc


# TODO logging
def multicut_kernighan_lin(graph, costs, warmstart=True, time_limit=None):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_gaec(graph, costs, time_limit=None):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.greedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def key_to_agglomerator(key):
    agglo_dict = {'kernighan-lin': multicut_kernighan_lin,
                  'greedy-additive': multicut_gaec}
    assert key in agglo_dict, key
    return agglo_dict[key]
