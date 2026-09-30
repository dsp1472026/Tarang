"""
Dual-code recovery over GF(2) (blind parity-check estimation).

Given B aligned blocks of n bits that are codewords of an unknown binary linear code (rows of
Y, B > n), every parity check h satisfies Y h = 0, so the null space of Y is the dual code.
For LDPC codes (and for convolutional codes cut into blocks) the dual is spanned by SPARSE
checks, which are what a message-passing decoder needs. They are searched with an
information-set approach: the null-space basis is put in reduced row-echelon form over a
random column order; the rows (and pairwise sums of rows) of weight <= w_max are kept.
Repeating with new column orders collects many distinct low-weight checks.

This recovers a parity-check matrix H_est of the code as observed (after any permutation);
it needs error-free blocks (the null space of noisy data is empty) and more blocks than n.
"""

from typing import List, Optional

import numpy as np


def rref(M: np.ndarray, col_order: Optional[np.ndarray] = None):
    """Reduced row echelon form over GF(2). Returns (R, pivot_columns) in the ORIGINAL column
    indexing; col_order sets the order in which columns are tried as pivots."""
    A = (np.asarray(M, dtype=np.uint8) & 1).copy()
    m, n = A.shape
    order = np.arange(n) if col_order is None else np.asarray(col_order)
    pivots = []
    r = 0
    for c in order:
        if r == m:
            break
        rows = np.flatnonzero(A[r:, c]) + r
        if len(rows) == 0:
            continue
        p = rows[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        others = np.flatnonzero(A[:, c])
        others = others[others != r]
        if len(others):
            A[others] ^= A[r]
        pivots.append(int(c))
        r += 1
    return A[:r], pivots


def null_space(M: np.ndarray) -> np.ndarray:
    """Basis (rows) of {h : M h = 0} over GF(2)."""
    M = np.asarray(M, dtype=np.uint8) & 1
    n = M.shape[1]
    R, piv = rref(M)
    free = [c for c in range(n) if c not in set(piv)]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, pc in enumerate(piv):
            if R[r, f]:
                basis[i, pc] = 1
    return basis


def sparse_checks(basis: np.ndarray, w_max: int, iterations: int = 60, seed: int = 1,
                  pair_sums: bool = True) -> np.ndarray:
    """Distinct vectors of weight in [2, w_max] in the span of `basis` (rows)."""
    rng = np.random.default_rng(seed)
    d, n = basis.shape
    found = {}
    for it in range(iterations):
        order = rng.permutation(n)
        R, _ = rref(basis, order)
        cands = [R]
        if pair_sums and len(R) > 1:
            w = R.sum(axis=1)
            light = R[np.argsort(w)[:min(len(R), 40)]]
            i, j = np.triu_indices(len(light), 1)
            cands.append(light[i] ^ light[j])
        for C in cands:
            w = C.sum(axis=1)
            for row in C[(w >= 2) & (w <= w_max)]:
                found.setdefault(row.tobytes(), row.copy())
    if not found:
        return np.zeros((0, n), dtype=np.uint8)
    H = np.array(list(found.values()), dtype=np.uint8)
    return H[np.argsort(H.sum(axis=1), kind="stable")]


def independent_rows(H: np.ndarray) -> np.ndarray:
    """A maximal linearly independent subset of the rows of H (lightest first)."""
    keep, basis_rows = [], np.zeros((0, H.shape[1]), dtype=np.uint8)
    rank = 0
    for row in H:
        test = np.vstack([basis_rows, row])
        R, _ = rref(test)
        if len(R) > rank:
            basis_rows, rank = test, len(R)
            keep.append(row)
    return np.array(keep, dtype=np.uint8) if keep else np.zeros((0, H.shape[1]), dtype=np.uint8)


def recover_parity_checks(blocks: np.ndarray, w_max: int, iterations: int = 60) -> dict:
    """Blocks (B x n, B > n) -> sparse parity-check matrix estimate and statistics."""
    blocks = np.asarray(blocks, dtype=np.uint8) & 1
    basis = null_space(blocks)
    if len(basis) == 0:
        return {"dual_dimension": 0, "H": np.zeros((0, blocks.shape[1]), np.uint8), "row_weights": []}
    sp = sparse_checks(basis, w_max, iterations)
    H = independent_rows(sp) if len(sp) else sp
    return {"dual_dimension": int(len(basis)), "H": H, "row_weights": H.sum(axis=1).tolist() if len(H) else [],
            "complete": bool(len(H) == len(basis))}
