import nifty.graph.opt.multicut as nmc
import nifty.graph.opt.lifted_multicut as nlmc


# TODO enable time limit and logging
def multicut_kernighan_lin(graph, costs, warmstart=True):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    return solver.optimize()


def key_to_agglomerator(key):
    agglo_dict = {'kernighan-lin': multicut_kernighan_lin}
    assert key in agglo_dict, key
    return agglo_dict[key]
