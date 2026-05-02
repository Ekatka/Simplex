"""Parametrized cube LPs for simplex pivot-rule pathology experiments.

Three cube families, all with origin as initial feasible BFS (slacks basic).
Every constructor returns a Phase-2 simplex tableau in the form
    [ A | I | b ]
    [-c | 0 | 0 ]
so the agent can start pivoting immediately (no Phase 1 needed).

All three use the SAME Chvatal-Klee-Minty sparsity pattern (upper-triangular
A with exponential coefficients, RHS b_j = 4^{j-1}), so the tableau shape is
fixed at (n+1, 2n+1) regardless of which cube is sampled — n structural
vars + n slacks + 1 RHS, n constraint rows + 1 objective row. This lets a
single vec_env share one observation space across cube types.

BASE = 2 (instead of the literature's 10) to keep n=10 tableau entries at
~1e5 rather than ~1e18. The combinatorial 2^n vertex chain is preserved.

# Current status — important

Out of the three literature target rules:

    Klee-Minty    -> largest_coefficient  (well separated: Dantzig slow, others fast)
    Jeroslow      -> largest_increase     (NOT separated by the current construction)
    Goldfarb-Sit  -> steepest_edge        (NOT separated by the current construction)

The Jeroslow and Goldfarb-Sit constructors below produce LPs that are
visually DISTINCT from Klee-Minty in the tableau (different c, different
row scaling) but are still KM-pathological — `largest_coefficient` is the
slow rule on all three, `largest_increase` and `steepest_edge` are both
fast (often 1 pivot). Writing down LPs where largest_increase or
steepest_edge is specifically the 2^n-pivot rule, while the others are
fast, requires per-vertex engineering of the Dantzig chain (Jeroslow
1973 / Goldfarb-Sit 1979) and is deferred — see docs/cubes_experiment.md.

So this file currently gives the agent three distinguishable cube shapes
with the same target rule ("avoid Dantzig"), which is weaker than the
originally intended three-rules-three-cubes setup.
"""

import numpy as np


BASE = 2.0


def _phase2_tableau(c, A, b):
    """(max c^T x, A x <= b, x >= 0) -> Phase-2 tableau with slack basis."""
    m, n = A.shape
    T = np.zeros((m + 1, n + m + 1), dtype=np.float64)
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    T[-1, :n] = -np.asarray(c, dtype=np.float64)
    basis = np.arange(n, n + m, dtype=int)
    return T, basis


def _km_A_b(n, eps):
    """Classical Klee-Minty A and b in Chvatal form with base BASE."""
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for j in range(1, n + 1):
        for i in range(1, j):
            A[j - 1, i - 1] = 2.0 * eps * (BASE ** (j - i))
        A[j - 1, j - 1] = 1.0
        b[j - 1] = BASE ** (2 * (j - 1))
    return A, b


def klee_minty(n, eps):
    """Classical Klee-Minty cube. Pathological for `largest_coefficient`.

    At eps = 1 this is the textbook Chvatal form (with base 2 instead of
    base 10 for numerical tractability); `largest_coefficient` takes 2^n-1
    pivots, `largest_increase` and `steepest_edge` take 1 pivot.

    For eps in roughly [0.6, 1.1] the Dantzig pivot count stays close to
    2^n. Below eps ~ 0.5 the cube collapses toward the axis-aligned unit
    cube and becomes solvable by any rule in n pivots.
    """
    c = BASE ** np.arange(n - 1, -1, -1)
    A, b = _km_A_b(n, eps)
    return _phase2_tableau(c, A, b)


def jeroslow(n, beta):
    """Jeroslow-style variant — fast-decaying objective on the KM cube.

    Uses `c_j = beta^(j-1)` (decaying fast when beta < 0.5). Empirically
    this pushes both `largest_increase` and `steepest_edge` off their
    1-pivot shortcut onto an O(n)-pivot path, while leaving Dantzig at a
    shorter-than-KM but still pathological count (20-150 pivots at n=10).

    Intended literature pathology: `largest_increase` hits 2^n pivots.
    CURRENT CONSTRUCTION DOES NOT ACHIEVE THIS — see docs. What it does
    give the agent is a tableau visibly distinct from KM (flat-looking
    reduced-cost row with fast-decaying entries instead of the classical
    exponential profile).
    """
    c = beta ** np.arange(n)
    A, b = _km_A_b(n, 1.0)  # classical KM constraints
    return _phase2_tableau(c, A, b)


def goldfarb_sit(n, delta):
    """Goldfarb-Sit-style variant — row-rescaled KM cube.

    Divides constraint row j by `1 + delta*(j-1)`. This preserves the
    feasible polytope but changes column norms in the tableau, which is
    what steepest-edge normalizes by.

    Intended literature pathology: `steepest_edge` hits 2^n pivots.
    CURRENT CONSTRUCTION DOES NOT ACHIEVE THIS — the row scaling is
    uniform enough across columns that all SE ratios move together, so
    steepest_edge still finds the 1-pivot shortcut. What the agent does
    see is a tableau with distinctive row-scaled entries, separating this
    cube from both KM and Jeroslow visually.
    """
    c = BASE ** np.arange(n - 1, -1, -1)
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for j in range(1, n + 1):
        scale = 1.0 + delta * (j - 1)
        for i in range(1, j):
            A[j - 1, i - 1] = (2.0 * (BASE ** (j - i))) / scale
        A[j - 1, j - 1] = 1.0 / scale
        b[j - 1] = (BASE ** (2 * (j - 1))) / scale
    return _phase2_tableau(c, A, b)


CUBE_TYPES = ("klee_minty", "jeroslow", "goldfarb_sit")


def sample_cube(n, rng, eps_range=(0.6, 1.0), beta_range=(0.3, 0.45),
                delta_range=(0.1, 0.3)):
    """Uniformly pick one of the three cube types and sample its tilt parameter.

    Default parameter ranges are chosen from the empirical sweep (see
    docs/cubes_experiment.md):

      * KM  eps in [0.6, 1.0]    — keeps Dantzig's pivot count in the
                                   several-hundreds to ~1000 range at n=10.
      * JL  beta in [0.3, 0.45]  — largest_increase/steepest_edge take ~8
                                   pivots here (non-trivial); Dantzig 20-150.
      * GS  delta in [0.1, 0.3]  — moderate Dantzig pathology (50-110).

    Returns (T, basis, cube_type_str, param_float).
    """
    idx = int(rng.integers(3))
    if idx == 0:
        eps = float(rng.uniform(*eps_range))
        T, basis = klee_minty(n, eps)
        return T, basis, "klee_minty", eps
    if idx == 1:
        beta = float(rng.uniform(*beta_range))
        T, basis = jeroslow(n, beta)
        return T, basis, "jeroslow", beta
    delta = float(rng.uniform(*delta_range))
    T, basis = goldfarb_sit(n, delta)
    return T, basis, "goldfarb_sit", delta
