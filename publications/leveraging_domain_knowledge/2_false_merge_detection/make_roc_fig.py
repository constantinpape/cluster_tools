import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results():
    # with open('./roc_results_googlegt.json') as f:
    with open('./roc_results_fullgt_new.json') as f:
        results = json.load(f)
    return results


def plot_fixed_recall(recall):
    results = load_results()

    prec_values = []
    vi_merges_join = []
    vi_splits_join = []
    vi_merges_sep = []
    vi_splits_sep = []

    for key, result in results.items():
        rec, prec = key.split('_')
        rec, prec = float(rec), float(prec)
        if np.isclose(rec, recall):
            prec_values.append(prec)
            res = result[0]
            vi_merges_join.append(res['jointly']['vi-merge'])
            vi_splits_join.append(res['jointly']['vi-split'])
            vi_merges_sep.append(res['separately']['vi-merge'])
            vi_splits_sep.append(res['separately']['vi-split'])

    fig, axes = plt.subplots(2, constrained_layout=True)

    axes[0].plot(prec_values, vi_merges_sep)
    axes[0].set_xlabel('precision')
    axes[0].set_ylabel('VI-merge')

    axes[1].plot(prec_values, vi_splits_sep)
    axes[1].set_xlabel('precision')
    axes[1].set_ylabel('VI-split')

    fig.suptitle("Fixed recall: %f" % recall)
    plt.show()


def plot_fixed_precision(precision):
    results = load_results()

    rec_values = []
    vi_merges_join = []
    vi_splits_join = []
    vi_merges_sep = []
    vi_splits_sep = []

    for key, result in results.items():
        rec, prec = key.split('_')
        rec, prec = float(rec), float(prec)
        if np.isclose(prec, precision):
            rec_values.append(rec)
            res = result[0]
            vi_merges_join.append(res['jointly']['vi-merge'])
            vi_splits_join.append(res['jointly']['vi-split'])
            vi_merges_sep.append(res['separately']['vi-merge'])
            vi_splits_sep.append(res['separately']['vi-split'])

    fig, axes = plt.subplots(2, constrained_layout=True)

    axes[0].plot(rec_values, vi_merges_sep)
    axes[0].set_xlabel('recall')
    axes[0].set_ylabel('VI-merge')

    axes[1].plot(rec_values, vi_splits_sep)
    axes[1].set_xlabel('recall')
    axes[1].set_ylabel('VI-split')

    fig.suptitle("Fixed precision: %f" % precision)
    plt.show()


def mixed_plot():
    results = load_results()

    rec_values = []
    prec_values = []

    vi_merges_join = []
    vi_splits_join = []
    vi_merges_sep = []
    vi_splits_sep = []

    vi_merges_join_err = []
    vi_splits_join_err = []
    vi_merges_sep_err = []
    vi_splits_sep_err = []

    for key, result in results.items():
        rec, prec = key.split('_')
        rec, prec = float(rec), float(prec)
        if np.isclose(prec, rec):
            rec_values.append(rec)
            prec_values.append(prec)

            vi_join_split = [res['jointly']['vi-split'] for res in result]
            vi_join_merge = [res['jointly']['vi-merge'] for res in result]
            vi_sep_split = [res['separately']['vi-split'] for res in result]
            vi_sep_merge = [res['separately']['vi-merge'] for res in result]

            vi_splits_join.append(np.mean(vi_join_split))
            vi_merges_join.append(np.mean(vi_join_merge))
            vi_splits_join_err.append(np.std(vi_join_split))
            vi_merges_join_err.append(np.std(vi_join_merge))

            vi_splits_sep.append(np.mean(vi_sep_split))
            vi_merges_sep.append(np.mean(vi_sep_merge))
            vi_merges_sep_err.append(np.std(vi_sep_merge))
            vi_splits_sep_err.append(np.std(vi_sep_split))

    print("LMC-SI")
    print("Merge")
    print(vi_merges_sep)
    print("Split")
    print(vi_splits_sep)

    print("LMC-S")
    print("Merge")
    print(vi_merges_join)
    print("Split")
    print(vi_splits_sep)

    prec_values.append(1.)
    rec_values.append(1.)

    # values for google gt
    # vi_merges_join.append(0.2544)
    # vi_splits_join.append(1.3050)
    # vi_merges_join_err.append(0)
    # vi_splits_join_err.append(0)

    # vi_merges_sep.append(0.0122)
    # vi_splits_sep.append(1.2369)
    # vi_merges_sep_err.append(0)
    # vi_splits_sep_err.append(0)

    # vi_split_mc = 1.2189
    # vi_merge_mc = 0.6532

    # values for full gt
    vi_merges_join.append(0.9406)
    vi_splits_join.append(1.6110)
    vi_merges_join_err.append(0)
    vi_splits_join_err.append(0)

    vi_merges_sep.append(0.5403)
    vi_splits_sep.append(1.5773)
    vi_merges_sep_err.append(0)
    vi_splits_sep_err.append(0)

    vi_split_mc = 1.5246
    vi_merge_mc = 1.9057

    sns.set()
    fig, axes = plt.subplots(2, constrained_layout=True)

    axes[0].errorbar(rec_values, vi_merges_sep, yerr=vi_merges_sep_err, label='LMC-SI')
    axes[0].errorbar(rec_values, vi_merges_join, yerr=vi_merges_join_err, label='LMC-S')
    axes[0].plot(rec_values, [vi_merge_mc] * len(rec_values), label='MC')
    axes[0].set_xlabel('Oracle F-Score')
    axes[0].set_ylabel('VI-merge')
    axes[0].legend()

    axes[1].errorbar(rec_values, vi_splits_sep, yerr=vi_splits_sep_err, label='LMC-SI')
    axes[1].errorbar(rec_values, vi_splits_join, yerr=vi_splits_join_err, label='LMC-S')
    axes[1].plot(rec_values, [vi_split_mc] * len(rec_values), label='MC')
    axes[1].set_xlabel('Oracle F-Score')
    axes[1].set_ylabel('VI-split')
    axes[1].legend()

    # fig.suptitle('Tuning recall and precision')
    plt.show()


def make_fixed_plots():
    for val in (.5, .75, 1.):
        plot_fixed_recall(val)
        plot_fixed_precision(val)


if __name__ == '__main__':
    # make_fixed_plots()
    # plot_fixed_precision(.5)
    mixed_plot()
