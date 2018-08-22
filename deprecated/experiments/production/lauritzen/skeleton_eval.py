import os
import json
import numpy as np


def load_eval(ws_type,
              use_rf=False,
              use_lmc=False,
              weight_mc_edges=False,
              weight_merge_edges=False):
    cache_prefix = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/'

    block_id = 2
    cache_name_mc = 'lauritzen_production_%i' % block_id
    mc_string = 'lmc_' if use_lmc else 'mc_'
    mc_string += ws_type
    if use_rf:
        mc_string += '_rf'
    mc_string += '_mcweighted' if weight_mc_edges else '_nomcweighted'
    mc_string += '_mergeweighted' if weight_merge_edges else '_nomergeweighted'
    cache_folder_mc = os.path.join(cache_prefix, cache_name_mc + '_' + mc_string)

    eval_file = os.path.join(cache_folder_mc, 'skeleton_eval_res.json')
    if not os.path.exists(eval_file):
        return mc_string, None

    with open(eval_file) as f:
        eval_ = json.load(f)
    return mc_string, eval_


def compare():
    ws_types = ['ws_thresh', 'ws_dt']
    weight_mc_edges = [True, False]
    weight_merge_edges = [True, False]

    for ws_type in ws_types:
        for weight_mc in weight_mc_edges:
            for weight_merge in weight_merge_edges:
                descr, eval_ = load_eval(ws_type, weight_mc_edges=weight_mc,
                                         weight_merge_edges=weight_merge)
                if eval_ is None:
                    print("No eval present, skipping", descr)
                    continue

                print("Evaluation for:", descr)
                for key, val in eval_.items():
                    print(key)
                    print(val)
                print()


def single_plot(params, eval_key, ax, labels, title):
    assert len(labels) == len(params)
    splits = []
    merges = []
    for param in params:
        _, eval_ = load_eval(*params)
        splits.append(eval_[eval_key]['split'])
        merges.append(eval_[eval_key]['merge'])

    splits = 100 * np.array(splits)
    merges = 100 * np.array(merges)

    width = .35

    index = np.arange(len(labels))
    ax.bar(index, merges, width, color='r', label='false merges')
    ax.bar(index + width, splits, width, color='g', label='false splits')

    ax.set_ylabel("Incorrect skeleton edges in %")
    ax.set_xlabel("Multicut / Stitching method")
    ax.set_xticks(index)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend(loc=2)


def plot_eval():
    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2)
    fig, ax = plt.subplots(1)

    params = [('ws_thresh', True, True), ('ws_dt', True, False)]

    eval_key1 = 'skeletons/for_eval_20180523'
    t1 = 'Evaluation for all skeletons'
    plot_eval(params, eval_key1, ax[0], labels, t1)

    # eval_key2 = 'skeletons/neurons_of_interest'
    # keys2 = [key + '_' + postfix for key in keys]
    # t2 = 'Evaluation for skeletons of interest'
    # plot_eval(path, keys2, ax[1], labels, t2)

    plt.show()


if __name__ == '__main__':
    compare()
    quit()
    path = './eval_block2.json'
    keys = ['mc_glia_global2', 'mc_glia_affs', 'mc_glia_rf_affs_global', 'mc_glia_affs_rf']
    labels = ['affs-global', 'affs-local', 'rf-global', 'rf-local']
