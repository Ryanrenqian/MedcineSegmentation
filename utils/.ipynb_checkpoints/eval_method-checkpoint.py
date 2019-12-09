import numpy as np
import glob
import random

"""BreasPathQ的评测代码"""


def predprob(truth_labels, pred, initial_lexsort=True):
    """
    Calculates the prediction probability. Adapted from  scipy's implementation of Kendall's Tau

    Note: x should be the truth labels.

    Parameters
    ----------
    truth_labels, pred : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they will
        be flattened to 1-D.
    initial_lexsort : bool, optional
        Whether to use lexsort or quicksort as the sorting method for the
        initial sort of the inputs. Default is lexsort (True), for which
        `predprob` is of complexity O(n log(n)). If False, the complexity is
        O(n^2), but with a smaller pre-factor (so quicksort may be faster for
        small arrays).
    Returns
    -------
    Prediction probability : float

    Notes
    -----
    The definition of prediction probability that is used is:
      p_k = (((P - Q) / (P + Q + T)) + 1)/2
    where P is the number of concordant pairs, Q the number of discordant
    pairs, and T the number of ties only in `y`.
    References
    ----------
    Smith W.D, Dutton R.C, Smith N.T. (1996) A measure of association for assessing prediction accuracy
    that is a generalization of non-parametric ROC area. Stat Med. Jun 15;15(11):1199-215
    """

    truth_labels = np.asarray(truth_labels).ravel()
    pred = np.asarray(pred).ravel()

    if not truth_labels.size or not pred.size:
        return (np.nan, np.nan)  # Return NaN if arrays are empty

    n = np.int64(len(truth_labels))
    temp = list(range(n))  # support structure used by mergesort

    # this closure recursively sorts sections of perm[] by comparing
    # elements of y[perm[]] using temp[] as support
    # returns the number of swaps required by an equivalent bubble sort

    def mergesort(offs, length):
        exchcnt = 0
        if length == 1:
            return 0
        if length == 2:
            if pred[perm[offs]] <= pred[perm[offs + 1]]:
                return 0
            t = perm[offs]
            perm[offs] = perm[offs + 1]
            perm[offs + 1] = t
            return 1
        length0 = length // 2
        length1 = length - length0
        middle = offs + length0
        exchcnt += mergesort(offs, length0)
        exchcnt += mergesort(middle, length1)
        if pred[perm[middle - 1]] < pred[perm[middle]]:
            return exchcnt
        # merging
        i = j = k = 0
        while j < length0 or k < length1:
            if k >= length1 or (j < length0 and pred[perm[offs + j]] <=
                                pred[perm[middle + k]]):
                temp[i] = perm[offs + j]
                d = i - j
                j += 1
            else:
                temp[i] = perm[middle + k]
                d = (offs + i) - (middle + k)
                k += 1
            if d > 0:
                exchcnt += d
            i += 1
        perm[offs:offs + length] = temp[0:length]
        return exchcnt

    # initial sort on values of x and, if tied, on values of y
    if initial_lexsort:
        # sort implemented as mergesort, worst case: O(n log(n))
        perm = np.lexsort((pred, truth_labels))
    else:
        # sort implemented as quicksort, 30% faster but with worst case: O(n^2)
        perm = list(range(n))
        perm.sort(key=lambda a: (truth_labels[a], pred[a]))

    # compute joint ties
    first = 0
    t = 0
    for i in range(1, n):
        if truth_labels[perm[first]] != truth_labels[perm[i]] or pred[perm[first]] != pred[perm[i]]:
            t += ((i - first) * (i - first - 1)) // 2
            first = i
    t += ((n - first) * (n - first - 1)) // 2

    # compute ties in x
    first = 0
    u = 0
    for i in range(1, n):
        if truth_labels[perm[first]] != truth_labels[perm[i]]:
            u += ((i - first) * (i - first - 1)) // 2
            first = i
    u += ((n - first) * (n - first - 1)) // 2

    # count exchanges
    exchanges = mergesort(0, n)
    # compute ties in y after mergesort with counting
    first = 0
    v = 0
    for i in range(1, n):
        if pred[perm[first]] != pred[perm[i]]:
            v += ((i - first) * (i - first - 1)) // 2
            first = i
    v += ((n - first) * (n - first - 1)) // 2

    tot = (n * (n - 1)) // 2
    if tot == u or tot == v:
        return (np.nan, np.nan)  # Special case for all ties in both ranks

    p_k = (((tot - (v + u - t)) - 2.0 * exchanges) / (tot - u) + 1) / 2

    return p_k
