import numpy as np


def contigency_table(seg_a, seg_b):
    """ Compute the pairs and counts in the contingency table of seg_a and seg_b.

    The contingency table counts the number of pixels that are shared between
    objects from seg_a and seg_b
    """
    # compute the unique ids and couunts for seg a and seg b
    # and wrap them in a dict
    a_ids, a_counts = np.unique(seg_a, return_counts=True)
    b_ids, b_counts = np.unique(seg_b, return_counts=True)
    a_dict = dict(zip(a_ids, a_counts))
    b_dict = dict(zip(b_ids, b_counts))
    # compute the overlaps and overlap counts
    pairs = np.rec.fromrecords(zip(a_ids, b_ids))
    p_ids, p_counts = np.unique(pairs, return_counts=True)
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
    vis = sum_a - sum_ab
    vim = sum_b - sum_ab
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

    # compute ids, counts and overlaps making up the contigency table
    a_dict, b_dict, p_ids, p_counts = contigency_table(seg_a, seg_b)
    n_points = seg_a.size

    # compute and return vi scores
    vis, vim = compute_vi_scores(a_dict, b_dict, p_ids, p_counts, n_points,
                                 use_log2=use_log2)
    return vis, vim


# TODO
def compute_rand_scores():
    raise NotImplementedError("Rand not implemented yet")


# TODO
def rand_index(seg_a, seg_b, ignore_a=None, ignore_b=None):
    """ Compute rand index derived scores between to segmentations.

    Computes split and merge vi, add them up to get the full vi score

    Arguments:
        seg_a [np.ndarray] - segmentation a
        seg_b [np.ndarray] - segmentation b
        ignore_a [listlike] - ignore ids for segmentation a (default: None)
        ignore_b [listlike] - ignore ids for segmentation b (default: None)
    Retuns:
        TODO
    """
    raise NotImplementedError("Rand not implemented yet")
    ignore_mask = compute_ignore_mask(seg_a, seg_b, ignore_a, ignore_b)
    if ignore_mask is not None:
        seg_a = seg_a[ignore_mask]
        seg_b = seg_b[ignore_mask]

    # compute ids, counts and overlaps making up the contigency table
    a_dict, b_dict, p_ids, p_counts = contigency_table(seg_a, seg_b)
    # n_points = seg_a.size

    # compute and return vi scores
    compute_rand_scores()
    return
