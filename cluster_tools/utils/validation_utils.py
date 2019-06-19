import numpy as np
import nifty.ground_truth as ngt


#
# helper functions for contingency tables, masking and serialization
#

def contigency_table(seg_a, seg_b):
    """ Compute the pairs and counts in the contingency table of seg_a and seg_b.

    The contingency table counts the number of pixels that are shared between
    objects from seg_a and seg_b.
    """
    # compute the unique ids and couunts for seg a and seg b
    # and wrap them in a dict
    a_ids, a_counts = np.unique(seg_a, return_counts=True)
    b_ids, b_counts = np.unique(seg_b, return_counts=True)
    a_dict = dict(zip(a_ids, a_counts))
    b_dict = dict(zip(b_ids, b_counts))

    # compute the overlaps and overlap counts
    # use nifty gt functionality
    ovlp_comp = ngt.overlap(seg_a, seg_b)
    ovlps = [ovlp_comp.overlapArrays(ida, sorted=False) for ida in a_ids]
    p_ids = np.array([[ida, idb] for ida, ovlp in zip(a_ids, ovlps) for idb in ovlp[0]])
    p_counts = np.concatenate([ovlp[1] for ovlp in ovlps]).astype('uint64')
    assert len(p_ids) == len(p_counts)

    # the contingency table is unsorted.
    # it can be sorted with the code below, but is not necessary here
    sorted_ids = np.lexsort(np.rot90(p_ids))
    p_ids = p_ids[sorted_ids]
    p_counts = p_counts[sorted_ids]
    print(p_ids[:10])
    print(p_counts[:10])

    # nifty function overcounts by a factor of 2
    p_counts = np.divide(p_counts, 2)

    # this is the alternative (naive) numpy impl, unfortunately this is very slow and
    # needs a lot of memory
    # pairs = np.concatenate((seg_a[:, None], seg_b[:, None]), axis=1)
    # p_ids_, p_counts_ = np.unique(pairs, return_counts=True, axis=0)

    return a_dict, b_dict, p_ids, p_counts


def compute_ignore_mask(seg_a, seg_b, ignore_a, ignore_b):
    if ignore_a is None and ignore_b is None:
        return None
    ignore_mask_a = None if ignore_a is None else np.isin(seg_a, ignore_a)
    ignore_mask_b = None if ignore_b is None else np.isin(seg_b, ignore_b)

    if ignore_mask_a is not None and ignore_mask_b is None:
        ignore_mask = ignore_mask_a
    elif ignore_mask_a is None and ignore_mask_b is not None:
        ignore_mask = ignore_mask_b
    elif ignore_mask_a is not None and ignore_mask_b is not None:
        ignore_mask = np.logical_and(ignore_mask_a, ignore_mask_b)

    # need to invert the mask
    return np.logical_not(ignore_mask)


#
# vi metrics
#


def compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points, use_log2):

    log = np.log2 if use_log2 else np.log

    # compute the vi-primitves
    a_counts = a_dict.values()
    sum_a = sum(-c / n_points * log(c / n_points) for c in a_counts)

    b_counts = b_dict.values()
    sum_b = sum(-c / n_points * log(c / n_points) for c in b_counts)

    sum_ab = np.sum([c / n_points * log(n_points * c / (a_dict[a] * b_dict[b]))
                     for (a, b), c in zip(p_ids, p_counts)])
    # compute the actual vi-scores (split-vi, merge-vi)
    vis = sum_b - sum_ab
    vim = sum_a - sum_ab
    return vis, vim


def variation_of_information(seg_a, seg_b, ignore_a=None, ignore_b=None, use_log2=True):
    """ Compute variation of information between two segmentations.

    Computes split and merge vi, add them up to get the full vi score.

    Arguments:
        seg_a [np.ndarray] - segmentation a
        seg_b [np.ndarray] - segmentation b
        ignore_a [listlike] - ignore ids for segmentation a (default: None)
        ignore_b [listlike] - ignore ids for segmentation b (default: None)
        use_log2 [bool] - whether to use log2 or loge (default: True)
    Retuns:
        float - split vi
        float - merge vi
    """
    ignore_mask = compute_ignore_mask(seg_a, seg_b, ignore_a, ignore_b)
    if ignore_mask is not None:
        seg_a = seg_a[ignore_mask]
        seg_b = seg_b[ignore_mask]
    else:
        # if we don't have a mask, we need to make sure the segmentations are
        seg_a = seg_a.ravel()
        seg_b = seg_b.ravel()

    # compute ids, counts and overlaps making up the contigency table
    a_dict, b_dict, p_ids, p_counts = contigency_table(seg_a, seg_b)
    n_points = seg_a.size

    # compute and return vi scores
    vis, vim = compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points,
                                 use_log2=use_log2)
    return vis, vim


#
# rand metrics
#


def compute_rand_scores(a_dict, b_dict, p_counts, n_points):

    # compute the rand-primitves
    a_counts = a_dict.values()
    sum_a = float(sum(c * c for c in a_counts))

    b_counts = b_dict.values()
    sum_b = float(sum(c * c for c in b_counts))

    sum_ab = float(sum(c * c for c in p_counts))

    prec = sum_ab / sum_b
    rec = sum_ab / sum_a

    # compute rand scores:
    # adapted rand index and randindex
    ari = (2 * prec * rec) / (prec + rec)
    ri = 1. - (sum_a + sum_b - 2 * sum_ab) / (n_points * n_points)
    ari = 1. - ari

    return ari, ri


def rand_index(seg_a, seg_b, ignore_a=None, ignore_b=None):
    """ Compute rand index derived scores between two segmentations.

    Computes adapted rand error and rand index.

    Arguments:
        seg_a [np.ndarray] - segmentation a
        seg_b [np.ndarray] - segmentation b
        ignore_a [listlike] - ignore ids for segmentation a (default: None)
        ignore_b [listlike] - ignore ids for segmentation b (default: None)
    Retuns:
        float - adapted rand error
        float - rand index
    """
    ignore_mask = compute_ignore_mask(seg_a, seg_b, ignore_a, ignore_b)
    if ignore_mask is not None:
        seg_a = seg_a[ignore_mask]
        seg_b = seg_b[ignore_mask]
    else:
        # if we don't have a mask, we need to make sure the segmentations are
        seg_a = seg_a.ravel()
        seg_b = seg_b.ravel()

    # compute ids, counts and overlaps making up the contigency table
    a_dict, b_dict, _, p_counts = contigency_table(seg_a, seg_b)
    n_points = seg_a.size

    # compute and return rand scores
    ari, ri = compute_rand_scores(a_dict, b_dict, p_counts, n_points)
    return ari, ri


def cremi_score(seg_a, seg_b, ignore_a=None, ignore_b=None):
    """ Computes cremi scores between two segmentations

    Arguments:
        seg_a [np.ndarray] - segmentation a
        seg_b [np.ndarray] - segmentation b
        ignore_a [listlike] - ignore ids for segmentation a (default: None)
        ignore_b [listlike] - ignore ids for segmentation b (default: None)
    Retuns:
        float - vi-split
        float - vi-merge
        float - adapted rand error
        float - cremi score
    """

    ignore_mask = compute_ignore_mask(seg_a, seg_b, ignore_a, ignore_b)
    if ignore_mask is not None:
        seg_a = seg_a[ignore_mask]
        seg_b = seg_b[ignore_mask]
    else:
        # if we don't have a mask, we need to make sure the segmentations are
        seg_a = seg_a.ravel()
        seg_b = seg_b.ravel()

    # compute ids, counts and overlaps making up the contigency table
    a_dict, b_dict, p_ids, p_counts = contigency_table(seg_a, seg_b)
    n_points = seg_a.size

    # compute vi scores
    vis, vim = compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points,
                                 use_log2=True)

    # compute and rand scores
    ari, _ = compute_rand_scores(a_dict, b_dict, p_counts, n_points)

    # compute the cremi score = geometric mean of voi and ari
    cs = np.sqrt(ari * (vis + vim))

    return vis, vim, ari, cs
