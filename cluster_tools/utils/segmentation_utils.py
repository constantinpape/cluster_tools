from concurrent import futures
import vigra
import nifty
import nifty.ufd as nufd
import nifty.graph.opt.multicut as nmc
import nifty.graph.opt.lifted_multicut as nlmc


# TODO logging
def multicut_kernighan_lin(graph, costs, warmstart=True, time_limit=None, n_threads=1):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_gaec(graph, costs, time_limit=None, n_threads=1):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.greedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_decomposition(graph, costs, time_limit=None, n_threads=1):
    # merge attractive edges with ufd
    merge_edges = costs > 0
    ufd = nufd.ufd(graph.numberOfNodes)
    uv_ids = graph.uvIds()
    ufd.merge(uv_ids[merge_edges])
    cc_labels = ufd.elementLabeling()
    # relabel consecutive
    cc_labels, max_id, _ = vigra.analysis.relabelConsecutive(cc_labels, start_label=0,
                                                             keep_zeros=False)

    # TODO check that relabelConsecutive lifts gil ....
    # solve a component sub-problem
    def solve_component(component_id):
        nodes = np.where(cc_labels == component_id)[0]
        inner_edges, _, sub_uvs = graph.extractSubgraphFromNodes(nodes)
        # relabel the local graph
        sub_uvs, n_local_nodes, _ = vigra.analysis.relabelConsecutive(sub_uvs)
        sub_graph = nifty.undirectedGraph(n_local_nodes + 1)
        sub_graph.insertEdges(sub_uvs)
        sub_labels = multicut_kernighan_lin(sub_graph, costs[inner_edges])
        sub_labels, max_id, _ = vigra.analysis.relabelConsecutive(sub_labels, start_label=0,
                                                                  keep_zeros=False)
        return sub_labels, max_id + 1

    # solve all components in parallel
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(solve_component, component_id)
                 for component_id in range(max_id + 1)]
        results = [t.result() for t in tasks]

    sub_results = [res[0] for res in results]
    offsets = np.array([res[1] for res in results], dtype='uint64')

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)

    # insert subsolutions into the components
    node_labels = np.zeros_like(cc_labels, dtype='uint64')

    # TODO not sure if the sub result node labels are in correct oreder
    def insert_solution(component_id):
        nodes = np.where(cc_labels == component_id)[0]
        node_labels[nodes] = (sub_results[component_id] + offsets[component_id])

    with futures.ThreadPoolExector(n_threads) as tp:
        tasks = [tp.submit(insert_solution, component_id)
                 for component_id in range(max_id + 1)]
        [t.result() for t in tasks]

    return node_labels


def key_to_agglomerator(key):
    agglo_dict = {'kernighan-lin': multicut_kernighan_lin,
                  'greedy-additive': multicut_gaec,
                  'decomposition': multicut_decomposition}
    assert key in agglo_dict, key
    return agglo_dict[key]
