import numpy as np
import nifty
import nifty.graph.rag as nrag
import nifty.graph.opt.lifted_multicut as nlmc
import cremi_tools.segmentation as cseg
from production import features as feat


##
# multicut functions
##


# we refactor this further to make it possible to
# call from compute_mc and compute_lmc
def run_mc(graph, probs, uv_ids,
           edge_sizes=None,
           weighting_exponent=None,
           with_ignore_edges=True):

    # build multicut solver
    mc = cseg.Multicut('kernighan-lin', weight_edges=edge_sizes is not None)

    # if we still have edges to ignore label, filter them (if we come here from lifted mc, we don't)
    # set edges connecting to 0 (= ignore label) to repulsive
    # (implemented like this, because we don't need to filter if we
    # come here from LMC workflow)
    if with_ignore_edges:
        ignore_edges = (uv_ids == 0).any(axis=1)
        # if we have edge sizes, set them to 1 for ignore edges,
        # to not skew the max calculation
        if edge_sizes is not None:
            edge_sizes[ignore_edges] = 1

    # transform probabilities to costs
    if edge_sizes is not None:
        costs = mc.probabilities_to_costs(probs, edge_sizes=edge_sizes,
                                          weighting_exponent=weighting_exponent)
    else:
        costs = mc.probabilities_to_costs(probs)

    if with_ignore_edges:
        costs[ignore_edges] = -100

    # solve the mc problem
    node_labels = mc(graph, costs)

    # get indicators for merge !
    # and return uv-ids, edge indicators and edge sizes
    merge_indicator = (node_labels[uv_ids[:, 0]] == node_labels[uv_ids[:, 1]]).astype('uint8')
    return node_labels, merge_indicator


def compute_mc_learned(ws, affs, offsets,
                       n_labels, weight_mulitcut_edges,
                       weighting_exponent, rf):
    assert len(rf) == 2
    # compute the region adjacency graph
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=1)
    uv_ids = rag.uvIds()
    if uv_ids.size == 0:
        return None, None, None

    # TODO add glia features ?
    rf_xy, rf_z = rf
    features, sizes, z_edges = feat.edge_features(rag, ws, n_labels, uv_ids, affs[:3])

    probs = np.zeros(len(features))
    xy_edges = np.logical_not(z_edges)
    if np.sum(xy_edges) > 0:
        probs[xy_edges] = rf_xy.predict_proba(features[xy_edges])[:, 1]
    if np.sum(z_edges) > 0:
        probs[z_edges] = rf_z.predict_proba(features[z_edges])[:, 1]

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    # compute multicut edge results
    _, merge_indicator = run_mc(graph, probs, uv_ids,
                                with_ignore_edges=True,
                                edge_sizes=sizes if weight_mulitcut_edges else None,
                                weighting_exponent=weighting_exponent)

    return uv_ids, merge_indicator, sizes


def compute_mc(ws, affs, offsets,
               n_labels, weight_mulitcut_edges,
               weighting_exponent):
    # compute the region adjacency graph
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=1)
    uv_ids = rag.uvIds()
    if uv_ids.size == 0:
        return None, None, None

    # compute the features and get edge probabilities (from mean affinities)
    # and edge sizes
    features = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets,
                                                       numberOfThreads=1)
    probs = features[:, 0]
    sizes = features[:, -1]

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    # compute multicut edge results
    _, merge_indicator = run_mc(graph, probs, uv_ids,
                                with_ignore_edges=True,
                                edge_sizes=sizes if weight_mulitcut_edges else None,
                                weighting_exponent=weighting_exponent)

    return uv_ids, merge_indicator, sizes


def run_lmc(graph, local_uv_ids, lifted_uv_ids, local_probs, lifted_probs,
            local_sizes=None, lifted_sizes=None, weighting_exponent=1., with_ignore_edges=True):

    if with_ignore_edges:
        # set the size of ignore edges to 1, to not mess up the weighting
        ignore_edges = (local_uv_ids == 0).any(axis=1)
        lifted_ignore_edges = (lifted_uv_ids == 0).any(axis=1)
        if local_sizes is not None:
            local_sizes[ignore_edges] = 1
        if lifted_sizes is not None:
            lifted_sizes[lifted_ignore_edges] = 1

    # turn probabilities into costs
    local_costs = cseg.transform_probabilities_to_costs(local_probs,
                                                        edge_sizes=local_sizes,
                                                        weighting_exponent=weighting_exponent)
    lifted_costs = cseg.transform_probabilities_to_costs(lifted_probs,
                                                         edge_sizes=lifted_sizes,
                                                         weighting_exponent=weighting_exponent)

    if with_ignore_edges:
        # set ignore labels (connecting to node id 0) to be maximally repulsive
        local_costs[ignore_edges] = -100
        lifted_costs[lifted_ignore_edges] = -100

    # build the lmc objective
    lifted_objective = nlmc.liftedMulticutObjective(graph)
    lifted_objective.setCosts(local_uv_ids, local_costs)
    lifted_objective.setCosts(lifted_uv_ids, lifted_costs)

    # compute lifted multicut
    solver_ga = lifted_objective.liftedMulticutGreedyAdditiveFactory().create(lifted_objective)
    node_labels = solver_ga.optimize()
    solver_kl = lifted_objective.liftedMulticutKernighanLinFactory().create(lifted_objective)
    node_labels = solver_kl.optimize(node_labels)

    # get indicators for merge !
    # and return uv-ids, edge indicators and edge sizes
    merge_indicator = (node_labels[local_uv_ids[:, 0]] == node_labels[local_uv_ids[:, 1]]).astype('uint8')
    return node_labels, merge_indicator


def compute_lmc_learned(ws, affs, glia,
                        offsets, n_labels, lifted_rf, lifted_nh,
                        weight_mulitcut_edges, weighting_exponent):
    # compute the region adjacency graph
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=1)
    uv_ids = rag.uvIds()

    # compute the features and get edge probabilities (from mean affinities)
    # and edge sizes
    features = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets,
                                                       numberOfThreads=1)
    local_probs = features[:, 0]
    sizes = features[:, -1].astype('uint64')

    # remove all edges connecting to the ignore label, because
    # they introduce short-cut lifted edges
    valid_edges = (uv_ids != 0).all(axis=1)
    uv_ids = uv_ids[valid_edges]

    # if we only had a single edge to ignore label, we can end up
    # with empty uv-ids at this point.
    # if so, return None
    if uv_ids.size == 0:
        return None, None, None

    local_probs = local_probs[valid_edges]
    sizes = sizes[valid_edges]

    # build the original graph and lifted objective
    # with lifted uv-ids
    lifted_uv_ids = feat.make_filtered_lifted_nh(rag, n_labels, uv_ids, lifted_nh)
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)

    # we may not get any lifted edges, in this case, fall back to normal multicut
    if lifted_uv_ids.size == 0:
        _, merge_indicator = run_mc(graph, local_probs, uv_ids,
                                    weighting_exponent=weighting_exponent,
                                    edge_sizes=sizes if weight_mulitcut_edges else None)
        return uv_ids, merge_indicator, sizes

    # get features for the lifted edges
    lifted_feats = np.concatenate([  # feat.ucm_features(n_labels, lifted_objective, local_probs),
                                   feat.clustering_features(graph, local_probs, lifted_uv_ids),
                                   feat.ucm_features(n_labels, uv_ids, lifted_uv_ids, local_probs),
                                   feat.region_features(ws, lifted_uv_ids, glia)], axis=1)
    lifted_probs = lifted_rf.predict_proba(lifted_feats)[:, 1]

    _, merge_indicator = run_lmc(graph, uv_ids, lifted_uv_ids, local_probs, lifted_probs,
                                 local_sizes=sizes if weight_mulitcut_edges else None,
                                 weighting_exponent=weighting_exponent,
                                 with_ignore_edges=False)

    # we don't weight, because we might just have few lifted edges
    # and this would downvote the local edges significantly
    # weight the costs
    # n_local, n_lifted = len(uv_ids), len(lifted_uv_ids)
    # total = float(n_lifted) + n_local
    # local_costs *= (n_lifted / total)
    # lifted_costs *= (n_local / total)

    return uv_ids, merge_indicator, sizes


def compute_lmc(ws, affs,
                offsets, n_labels,
                weight_mulitcut_edges,
                weighting_exponent):
    # compute the region adjacency graph
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=1)
    uv_ids = rag.uvIds()

    # compute the lifted uv-ids, and features for local and lifted edges
    # from the affinities
    lifted_uvs, local_feats, lifted_feats = nrag.computeFeaturesAndNhFromAffinities(rag, affs, offsets,
                                                                                    numberOfThreads=1)

    local_probs = local_feats[:, 0]
    local_sizes = local_feats[:, -1].astype('uint64')

    lifted_probs = local_feats[:, 0]
    lifted_sizes = lifted_feats[:, -1].astype('uint64')

    # build the original graph and lifted objective
    # with lifted uv-ids
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)

    # we may not get any lifted edges, in this case, fall back to normal multicut
    if lifted_uvs.size == 0:
        merge_indicator = run_mc(graph, local_probs, uv_ids,
                                 weighting_exponent=weighting_exponent,
                                 edge_sizes=local_sizes if weight_mulitcut_edges else None,
                                 with_ignore_edges=True)
        return uv_ids, merge_indicator, local_sizes

    # we don't weight, because we might just have few lifted edges
    # and this would downvote the local edges significantly
    # weight the costs
    # n_local, n_lifted = len(uv_ids), len(lifted_uv_ids)
    # total = float(n_lifted) + n_local
    # local_costs *= (n_lifted / total)
    # lifted_costs *= (n_local / total)

    _, merge_indicator = run_lmc(graph, uv_ids, lifted_uvs,
                                 local_probs, lifted_probs,
                                 local_sizes=local_sizes if weight_mulitcut_edges else None,
                                 lifted_sizes=lifted_sizes if weight_mulitcut_edges else None,
                                 weighting_exponent=weighting_exponent,
                                 with_ignore_edges=True)
    return uv_ids, merge_indicator, local_sizes
