import numpy as np
import itertools
from typing import Dict, Tuple

def compute_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def compute_info_leak(data, bins=50):
    """
    data: shape (N, d)  where:
        data[:,0] = target T
        data[:,1:] = agents A1, A2, ...
    """
    T = data[:, 0]
    A = data[:, 1:]

    # Histogram estimate for T
    p_T, _ = np.histogram(T, bins=bins, density=True)
    p_T = p_T / np.sum(p_T)

    # Histogram estimate for joint (T,A)
    # we flatten A to a tuple of bins
    joint_bins = [bins] * (1 + A.shape[1])
    p_TA, _ = np.histogramdd(data, bins=joint_bins, density=True)
    p_TA = p_TA / np.sum(p_TA)

    # p(A)
    p_A = np.sum(p_TA, axis=0)
    # p(T|A)
    with np.errstate(divide='ignore', invalid='ignore'):
        p_T_given_A = p_TA / p_A[np.newaxis, :]

    # H(T)
    H_T = compute_entropy(p_T)

    # H(T|A)
    H_T_given_A = np.nansum(p_A * compute_entropy(p_T_given_A.reshape(bins, -1)))

    return H_T_given_A / H_T


def surd_global(MI: Dict[tuple, float], n_vars: int):
    """
    Full SURD decomposition using MI values from MLP estimator.
    Reproduces exactly the global SURD (Figure 13).

    Input:
      MI[(1,)] , MI[(2,)], ... , MI[(1,2)] , ...

    Output:
      I_R: redundant + unique
      I_S: synergy
      MI: original MI dictionary
    """

    # ---- Step 1: organize MI's by order ----
    T_sets = {}
    for comb, val in MI.items():
        k = len(comb)
        if k not in T_sets:
            T_sets[k] = []
        T_sets[k].append((comb, val))

    # ---- Step 2: sort each T̃M ascending ----
    for k in T_sets.keys():
        T_sets[k] = sorted(T_sets[k], key=lambda x: x[1])  # ascending MI

    # Prepare output
    I_R = {comb: 0.0 for comb in MI.keys()}
    I_S = {comb: 0.0 for comb in MI.keys() if len(comb) > 1}

    # ---- Step 3: Compute Redundant + Unique (only in T̃1) ----
    T1 = T_sets[1]           # list of (comb, val)
    n1 = len(T1)

    # convert to arrays
    I1_vals = np.array([v for (_, v) in T1])
    I1_combs = [c for (c, _) in T1]

    # I_0 = 0 for difference base
    prev = 0.0
    for i in range(n1):
        comb = I1_combs[i]
        val = I1_vals[i]
        diff = val - prev

        if i < n1 - 1:
            # Redundant
            I_R[comb] += diff
        else:
            # Unique
            I_R[comb] += diff

        prev = val

    # ---- Step 4: Higher-order synergy (T̃2, T̃3,...) ----
    for M in range(2, n_vars + 1):
        TM = T_sets.get(M, [])
        if len(TM) == 0:
            continue

        IM_vals = np.array([v for (_, v) in TM])
        IM_combs = [c for (c, _) in TM]

        # max of previous order
        prev_max = max([v for (_, v) in T_sets[M - 1]])

        prev = 0.0
        for i in range(len(TM)):
            comb = IM_combs[i]
            val = IM_vals[i]

            if prev >= prev_max:
                diff = val - prev
            else:
                if val > prev_max:
                    diff = val - prev_max
                else:
                    diff = 0.0

            I_S[comb] += max(diff, 0.0)
            prev = val

    return I_R, I_S, MI
