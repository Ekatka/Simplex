import gymnasium as gym
import numpy as np
from warnings import warn
import scipy.sparse as sps
from collections import namedtuple
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv

from _linprog_utils import (
    _parse_linprog, _presolve, _get_Abc, _LPProblem, _autoscale,
    _postsolve, _check_result, _display_summary)

from config import PIVOT_MAP, NUM_PIVOT_STRATEGIES

class SecondPhasePivotingEnv(gym.Env):
    def remove_artificial(self):
        for pivrow in [row for row in range(self.basis.size)
                       if self.basis[row] > self.T.shape[1] - 2]:
            non_zero_row = [col for col in range(self.T.shape[1] - 1)
                            if abs(self.T[pivrow, col]) > self.tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(self.T, self.basis, pivrow, pivcol, self.tol)
                self.nit += 1

    def __init__(self, T, basis):
        # Core state
        self.basis = basis
        self.T = T
        self.tol = 1e-9
        self.m = self.T.shape[1] - 1
        self.remove_artificial()

        # Limits & counters
        self.maxiter = 20_000
        self.nit = 0

        # Solution buffer
        if len(self.basis[:self.m]) == 0:
            self.solution = np.empty(self.T.shape[1] - 1, dtype=np.float64)
        else:
            self.solution = np.empty(max(self.T.shape[1] - 1, max(self.basis[:self.m]) + 1),
                                     dtype=np.float64)

        # Gym spaces
        self.action_space = spaces.Discrete(NUM_PIVOT_STRATEGIES)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.T.shape, dtype=np.float64
        )
        self.complete = False

        # === New: loop & progress bookkeeping ===
        self._seen_bases = set()
        self._last_obj = float(self.T[-1, -1])
        self._degenerate_streak = 0

        # === Tunable reward coefficients (fixed numbers, no placeholders) ===
        self._step_penalty = 1.0          # per-step cost
        self._improve_coef = 10.0         # scales positive objective improvement
        self._degenerate_penalty = 0.2    # small hit on zero-improvement pivot
        self._loop_penalty = 5.0          # larger penalty on detected cycle
        self._success_bonus = 50.0        # terminal reward on optimality
        self._improve_tol = 1e-12         # what counts as "no improvement"

    def _basis_key(self):
        # Small, stable key for visited-state detection
        return tuple(int(i) for i in self.basis)

    def _get_obs(self):
        return self.T.copy()

    def reset(self, seed=None, **kwargs):
        self.nit = 0
        self._seen_bases.clear()
        self._last_obj = float(self.T[-1, -1])
        self._degenerate_streak = 0
        # Record initial basis
        self._seen_bases.add(self._basis_key())
        return self._get_obs(), {}

    def step(self, action):
        strategy = PIVOT_MAP[int(action)]

        # Base reward: per-step penalty
        reward = -self._step_penalty
        done = False
        truncated = False

        # === Check optimality BEFORE pivot (no entering column) ===
        pivcol_found, pivcol = _pivot_col_heuristics(self.T, strategy=strategy, tol=self.tol)
        if not pivcol_found:
            # Optimal for Phase 2 (no negative reduced costs)
            reward += self._success_bonus
            done = True
            info = {
                "status": "optimal",
                "nit": self.nit,
                "objective": float(self.T[-1, -1]),
                "degenerate_streak": self._degenerate_streak,
            }
            return self._get_obs(), reward, done, truncated, info

        # === Choose leaving row ===
        use_bland = (strategy == 'blands_rule')
        pivrow_found, pivrow = _pivot_row(self.T, self.basis, pivcol, phase=2, tol=self.tol, bland=use_bland)
        if not pivrow_found:
            # Unbounded or infeasible in Phase 2 context
            done = True
            info = {
                "status": "no_pivot_row",
                "nit": self.nit,
                "objective": float(self.T[-1, -1]),
                "degenerate_streak": self._degenerate_streak,
            }
            return self._get_obs(), reward, done, truncated, info

        # === Apply pivot ===
        old_obj = float(self.T[-1, -1])
        _apply_pivot(self.T, self.basis, pivrow, pivcol, tol=self.tol)
        self.nit += 1
        new_obj = float(self.T[-1, -1])

        # === Reward shaping: improvement in objective (we minimize -v) ===
        # Improvement when new_obj < old_obj → delta positive
        delta = old_obj - new_obj
        if delta > self._improve_tol:
            reward += self._improve_coef * delta
            self._degenerate_streak = 0
        else:
            # Degenerate pivot (no progress)
            reward -= self._degenerate_penalty
            self._degenerate_streak += 1

        # === Loop detection on basis ===
        key = self._basis_key()
        if key in self._seen_bases:
            print("LOOP DETECTED")
            reward -= self._loop_penalty
            truncated = True  # cut the episode to break the cycle
        else:
            self._seen_bases.add(key)

        # === Hard stop on iteration cap ===
        if self.nit >= self.maxiter:
            done = True

        info = {
            "status": "running",
            "nit": self.nit,
            "objective": new_obj,
            "delta_objective": delta,
            "degenerate": (delta <= self._improve_tol),
            "degenerate_streak": self._degenerate_streak,
            "pivcol": int(pivcol),
            "pivrow": int(pivrow),
            "strategy": strategy,
            "loop_detected": truncated,
        }
        return self._get_obs(), reward, done, truncated, info

    def render(self, mode='human'):
        print("Current Tableau:")
        print(self.T)
        print("Current Basis:", self.basis)
        print("Iterations:", self.nit)

    def close(self):
        pass



class FirstPhasePivotingEnv(gym.Env):

    def __init__(self, T, basis, nit0 = 0):
        self.basis = basis
        self.T = T
        self.tol = 1e-9
        self.maxiter = 5000
        self.nit0 = nit0
        self.nit = 0
        self.action_space = spaces.Discrete(NUM_PIVOT_STRATEGIES)  # Use constant from config
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.T.shape, dtype=np.float64
        )
        self.complete = False

    def _get_obs(self):
        return self.T.copy()

    def reset(self, seed=None, **kwargs):
        self.nit = 0
        self.status = None
        return self._get_obs(), {}

    def step(self, action):
        # according to the given strategy find pivot column and row and apply the pivot
        # Use PIVOT_MAP from config instead of local definition
        strategy = PIVOT_MAP[int(action)]
        reward = -1
        done = False

        pivcol_found, pivcol = _pivot_col_heuristics(self.T, strategy=strategy, tol=self.tol)
        if not pivcol_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        pivrow_found, pivrow = _pivot_row(self.T, self.basis, pivcol, phase=1, tol=self.tol)
        if not pivrow_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        _apply_pivot(self.T, self.basis, pivrow, pivcol, tol=self.tol)
        self.nit += 1

        if self.nit >= self.maxiter:
            done = True

        # return reward -1 for each step
        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human'):
        print("Current Tableau:")
        print(self.T)
        print("Current Basis:", self.basis)
        print("Iteration:", self.nit)

    def close(self):
        pass



def potential_increase(T, col_index, cost_j, tol=1e-9):

    feasible_increases = []
    # We skip the last row (objective row), so iterate up to T.shape[0]-1
    for i in range(T.shape[0] - 1):
        pivot_val = T[i, col_index]
        if pivot_val > tol:
            ratio = T[i, -1] / pivot_val
            # cost_j is negative => improvement = -cost_j * ratio
            # (since cost_j < 0, -cost_j is positive => potential improvement)
            inc = -cost_j * ratio
            feasible_increases.append(inc)

    if feasible_increases:
        return max(feasible_increases)
    else:
        return None

def pivot_col_largest_increase(T, ma, tol=1e-9):
    cost_row = T[-1, :-1]  # objective coefficients excluding the RHS

    # Among negative entries, find the column with largest improvement
    col_candidates = np.ma.nonzero(ma < 0)[0]
    best_increase = -float('inf')
    best_col = None

    for j in col_candidates:
        cost_j = cost_row[j]
        inc = potential_increase(T, j, cost_j, tol=tol)
        if inc is not None and inc > best_increase:
            best_increase = inc
            best_col = j

    if best_col is None:
        # Means for every negative cost_j, no row was feasible => unbounded
        return False, np.nan

    return True, best_col

def _pivot_col_heuristics(T, strategy, tol=1e-9):
    """
    Pick pivot column using one of the supported heuristics.
    """
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan

    if strategy == 'blands_rule':
        # Bland's rule: choose the leftmost negative coefficient
        col = np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0]
        return True, col

    elif strategy == 'largest_coefficient':
        col = np.ma.nonzero(ma == ma.min())[0][0]
        return True,  col

    elif strategy == 'largest_increase':
        res, col = pivot_col_largest_increase(T, ma, tol)
        return res, col

    elif strategy == 'steepest_edge':
        cost_row = T[-1, :-1]
        best_ratio = -np.inf
        best_col = None
        col_candidates = np.ma.nonzero(ma < 0)[0]  # j where cost_j < -tol

        for j in col_candidates:
            cost_j = cost_row[j]
            col_vector = T[:-1, j]  # exclude the objective row
            col_norm = np.linalg.norm(col_vector)

            if col_norm <= tol:
                continue  # avoid division by near-zero or zero norm

            ratio = abs(cost_j) / col_norm

            if ratio > best_ratio:
                best_ratio = ratio
                best_col = j

        if best_col is None:
            return False, np.nan
        return True, best_col

def _pivot_col(T, tol=1e-9, bland=False):

    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan
    if bland:
        # ma.mask is sometimes 0d
        return True, np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0]
    return True, np.ma.nonzero(ma == ma.min())[0][0]


# def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):

#     if phase == 1:
#         k = 2
#     else:
#         k = 1
#     ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
#     if ma.count() == 0:
#         return False, np.nan
#     mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
#     q = mb / ma
#     min_rows = np.ma.nonzero(q == q.min())[0]
#     if bland:
#         return True, min_rows[np.argmin(np.take(basis, min_rows))]
#     return True, min_rows[0]
def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):
    """
    Harris two-pass ratio test for selecting the leaving row.

    Parameters
    ----------
    T : ndarray
        Simplex tableau.
    basis : ndarray
        Current basis indices (used only for Bland tie-break if requested).
    pivcol : int
        Entering column index.
    phase : int
        1 or 2; controls how many bottom rows are excluded from the ratio test.
    tol : float
        Positivity tolerance for the pivot column entries.
    bland : bool
        If True, tie-break among near-minimum ratios using smallest basis index.
        If False (default), choose the row with the largest pivot in the admissible set.

    Returns
    -------
    (found: bool, pivrow: int)
    """
    # Exclude objective rows at bottom (same as your original code)
    k = 2 if phase == 1 else 1
    col = T[:-k, pivcol]

    # Eligible rows have strictly positive pivot column entries
    eligible = col > tol
    if not np.any(eligible):
        return False, np.nan

    b = T[:-k, -1]

    # Compute ratios only for eligible rows
    q = np.full_like(col, np.inf, dtype=np.float64)
    q[eligible] = b[eligible] / col[eligible]

    # Pass 1: find robust minimum ratio
    q_min = np.min(q)
    if not np.isfinite(q_min):
        return False, np.nan

    # Pass 2: define admissible set near the minimum
    # eta: small relative tolerance; you can tweak 1e-7..1e-4 if needed
    eta = 1e-7
    threshold = (1.0 + eta) * q_min

    admissible = (q <= threshold) & np.isfinite(q)
    rows = np.where(admissible)[0]
    if rows.size == 0:
        # Fallback: strict minimizer (should rarely happen)
        rows = np.array([int(np.argmin(q))])

    if bland:
        # Bland tie-break among admissible rows: smallest basis index
        idx = rows[np.argmin(np.take(basis, rows))]
    else:
        # Choose numerically strongest pivot among admissible rows:
        # largest pivot element in the entering column
        pivots = col[rows]
        idx = rows[np.argmax(pivots)]

    return True, int(idx)


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):

    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] = T[pivrow] / pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] = T[irow] - T[pivrow] * T[irow, pivcol]

    # The selected pivot should never lead to a pivot value less than the tol.
    if np.isclose(pivval, tol, atol=0, rtol=1e4):
        # print("\n" + "="*80)
        # print("WARNING: NUMERICAL STABILITY ISSUE DETECTED")
        # print("="*80)
        
        # print(f"\nPIVOT OPERATION DETAILS:")
        # print(f"  • Pivot value: {pivval:.6e}")
        # print(f"  • Tolerance: {tol:.6e}")
        # print(f"  • Ratio (pivot/tolerance): {pivval/tol:.2f}")
        # print(f"  • Pivot row: {pivrow}")
        # print(f"  • Pivot column: {pivcol}")
        
        # print(f"\nTABLEAU INFORMATION:")
        # print(f"  • Tableau shape: {T.shape}")
        # print(f"  • Number of rows: {T.shape[0]}")
        # print(f"  • Number of columns: {T.shape[1]}")
        
        # print(f"\nBASIS INFORMATION:")
        # print(f"  • Current basis: {basis}")
        # print(f"  • Basis size: {len(basis)}")
        
        # print(f"\nPIVOT ELEMENT CONTEXT:")
        # print(f"  • Pivot element value: {pivval:.6e}")
        # print(f"  • Pivot row before normalization:")
        # pivot_row_before = T[pivrow] * pivval  # Reconstruct original row
        # print(f"    {pivot_row_before}")
        
        # print(f"\nNEARBY ELEMENTS (same row):")
        # for col in range(T.shape[1]):
        #     if col != pivcol:
        #         val = T[pivrow, col] * pivval  # Reconstruct original value
        #         if abs(val) > tol/10:  # Show elements close to tolerance
        #             print(f"    Column {col}: {val:.6e}")
        
        # print(f"\nNEARBY ELEMENTS (same column):")
        # for row in range(T.shape[0]):
        #     if row != pivrow:
        #         val = T[row, pivcol]
        #         if abs(val) > tol/10:  # Show elements close to tolerance
        #             print(f"    Row {row}: {val:.6e}")
        
        # print("="*80)
        # print("END OF WARNING")
        # print("="*80 + "\n")
        
        message = (
            f"The pivot operation produces a pivot value of:{pivval: .1e}, "
            "which is only slightly greater than the specified "
            f"tolerance{tol: .1e}. This may lead to issues regarding the "
            "numerical stability of the simplex method. "
            "Removing redundant constraints, changing the pivot strategy "
            "via Bland's rule or increasing the tolerance may "
            "help reduce the issue.")
        warn(message, stacklevel=5)



def phase1solver(T, basis,
                   maxiter=1000, tol=1e-9, nit0=0):
    nit = nit0
    status = 0
    phase=1
    bland=True
    message = ''
    complete = False
    m = T.shape[1] - 2
    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1),
                            dtype=np.float64)

    while not complete:
        # Find the pivot column
        pivcol_found, pivcol = _pivot_col(T, tol, bland=True)
        if not pivcol_found:
            pivcol = np.nan
            pivrow = np.nan
            status = 0
            complete = True
        else:
            # Find the pivot row
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)
            if not pivrow_found:
                status = 3
                complete = True


        if not complete:
            if nit >= maxiter:
                # Iteration limit exceeded
                status = 1
                complete = True
            else:
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    return nit, status


def change_to_zero_sum(GameMatrix):
    m, n = GameMatrix.shape
    c = np.append(np.zeros(m), -1)
    A_ub = np.hstack([-GameMatrix.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.append(np.ones(m), 0).reshape(1, -1)
    b_eq = [1]
    bounds = [(0, None)] * m + [(None, None)]

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0=None, integrality=None)
    options = None

    lp, solver_options = _parse_linprog(lp, options, meth='simplex')
    tol = solver_options.get('tol', 1e-9)
    c0 = 0
    A, b, c, c0, x0 = _get_Abc(lp, c0)

    n, m = A.shape

    # All constraints must have b >= 0.
    is_negative_constraint = np.less(b, 0)
    A[is_negative_constraint] *= -1
    b[is_negative_constraint] *= -1

    # As all cons   traints are equality constraints the artificial variables
    # will also be basic variables.
    av = np.arange(n) + m
    basis = av.copy()

    row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
    row_objective = np.hstack((c, np.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    return T, basis, av

# change from first phase to second
def first_to_second(T, basis, av):
    status = 0
    # Adaptive tolerance based on matrix size
    m = len(basis) - len(av)  # number of original variables
    base_tol = 1e-9
    adaptive_tol = base_tol * max(1, m / 10)  # Increase tolerance for larger matrices
    
    if abs(T[-1, -1]) < adaptive_tol:
        # Remove the pseudo-objective row from the tableau
        T = T[:-1, :]
        # Remove the artificial variable columns from the tableau
        T = np.delete(T, av, 1)
    else:
        # Failure to find a feasible starting point
        status = 2
        messages = {0: "Optimization terminated successfully.",
                    1: "Iteration limit reached.",
                    2: "Optimization failed. Unable to find a feasible"
                       " starting point.",
                    3: "Optimization failed. The problem appears to be unbounded.",
                    4: "Optimization failed. Singular matrix encountered."}
        messages[status] = (
            "Phase 1 of the simplex method failed to find a feasible "
            "solution. The pseudo-objective function evaluates to "
            f"{abs(T[-1, -1]):.1e} "
            f"which exceeds the required tolerance of {adaptive_tol} for a solution to be "
            "considered 'close enough' to zero to be a basic solution. "
            "Consider increasing the tolerance to be greater than "
            f"{abs(T[-1, -1]):.1e}. "
            "If this tolerance is unacceptably large the problem may be "
            "infeasible."
        )

    if status == 0:
        # Phase 2
        # nit2, status = phase2solver(T, n, basis)
        return T, basis

    else:
        print("[solve_zero_sum] Phase 1 failed or LP is numerically unstable.")
        print(f"[solve_zero_sum] Pseudo-objective: {T[-1, -1]:.2e} vs adaptive_tol {adaptive_tol:.1e}, status={status}")
        return None

def change_to_zero_sum_direct_phase2(GameMatrix):
    """
    Convert a zero-sum game matrix directly to a canonical Phase 2 tableau
    without Phase 1 by constructing a trivial feasible BFS and then
    canonicalizing the tableau.
    Returns (T, basis, None).
    """
    m, n = GameMatrix.shape

    # Shift so B >= 0
    min_element = np.min(GameMatrix)
    K = max(0.0, -min_element + 1e-6)
    B = GameMatrix + K

    # Build standard-form A, b, c for:
    #  -B^T x + v + s = 0  (n rows)
    #   1^T x        = 1  (1 row)
    slack_matrix = np.eye(n)
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])     # n x (m+1+n)
    A_sum         = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])     # 1 x (m+1+n)
    A = np.vstack([A_constraints, A_sum])                                 # (n+1) x (m+1+n)

    b = np.zeros(n + 1)
    b[-1] = 1.0

    c = np.zeros(m + 1 + n)
    c[m] = -1.0   # minimize -v

    # Choose a provably feasible, nonsingular basis:
    #  - First n rows: take all slacks s_j basic (columns m+1 ... m+n)
    #  - Last row (sum): take x_0 basic (column 0)
    # This yields B = [[I_n, -B^T[:,0]]; [0,...,0, 1]], invertible,
    # and basic values x0=1, s = B^T[:,0] >= 0 (since B>=0).
    basis = np.empty(n + 1, dtype=int)
    basis[:n] = m + 1 + np.arange(n)  # all slacks
    basis[-1] = 0                     # x[0]

    # Canonicalize
    T, basis = build_phase2_tableau_canonical(A, b, c, basis)

    # No artificial variables in this path
    return T, basis, None


def build_phase2_tableau_canonical(A, b, c, basis, tol=1e-12):
    """
    Given equality-form constraints A x = b (with x >= 0), an objective c^T x,
    and a chosen basis 'basis' (one column index per row), return the canonical
    Phase-2 simplex tableau:
        [ I | B^{-1}N | B^{-1}b ]
        [ 0 |  r_N     |   z0   ]
    where r_N = c_N - c_B B^{-1} N, z0 = c_B^T B^{-1} b.
    Assumes 'basis' selects a nonsingular square submatrix B of A.

    Args:
        A : (m x n) ndarray
        b : (m,) ndarray
        c : (n,) ndarray
        basis : (m,) ndarray of int column indices
        tol : float, numerical guard

    Returns:
        T  : ((m+1) x (n+1)) ndarray, canonical tableau
        basis : unchanged, but now consistent with T’s identity block
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    basis = np.asarray(basis, dtype=int)

    m, n = A.shape
    if basis.size != m:
        raise ValueError("basis length must equal number of rows m")

    # Partition columns into basis and nonbasis
    B_cols = basis
    N_cols = np.array([j for j in range(n) if j not in set(B_cols)], dtype=int)

    # Extract blocks
    B = A[:, B_cols]              # (m x m)
    N = A[:, N_cols]              # (m x (n-m))
    c_B = c[B_cols]               # (m,)
    c_N = c[N_cols]               # (n-m,)

    # Invert B (or solve)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError as e:
        raise ValueError("Chosen basis is singular; pick a different feasible basis.") from e

    # Canonical blocks
    B_inv_N = B_inv @ N           # (m x (n-m))
    B_inv_b = B_inv @ b           # (m,)

    # Objective reduction
    r_N = c_N - c_B @ B_inv_N     # (n-m,)
    z0  = c_B @ B_inv_b           # scalar

    # Assemble the full tableau columns in original variable order
    # Start with zeros; we will fill basis and nonbasis locations.
    top_left = np.zeros((m, n))
    # Put identity in basis columns
    for i, col in enumerate(B_cols):
        top_left[i, col] = 1.0
    # Put B^{-1}N into nonbasis columns
    for j_in, col in enumerate(N_cols):
        top_left[:, col] = B_inv_N[:, j_in]

    # RHS
    rhs = B_inv_b.reshape(-1, 1)

    # Objective row in original variable order
    obj_row = np.zeros(n)
    # Basic reduced costs are zero by construction
    for j_in, col in enumerate(N_cols):
        obj_row[col] = r_N[j_in]

    # Build tableau [ A' | b' ; obj | z0 ]
    T = np.vstack([
        np.hstack([top_left, rhs]),
        np.hstack([obj_row, np.array([z0])])
    ])

    # Tiny cleanup: zero-out near-zeros for numerical neatness
    T[np.abs(T) < tol] = 0.0
    return T, basis

def build_trivial_bfs_zero_sum_game(GameMatrix):
    """
    Build a canonical Phase 2 tableau for the zero-sum game using the
    trivial pure strategy x = e_0 as a feasible BFS.
    Returns (T, basis).
    """
    m, n = GameMatrix.shape

    # Shift so B >= 0
    min_element = np.min(GameMatrix)
    K = max(0.0, -min_element + 1e-6)
    B = GameMatrix + K

    # Standard-form A, b, c as above
    slack_matrix = np.eye(n)
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])
    A_sum         = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])
    A = np.vstack([A_constraints, A_sum])

    b = np.zeros(n + 1)
    b[-1] = 1.0

    c = np.zeros(m + 1 + n)
    c[m] = -1.0

    # Same feasible, nonsingular basis: all slacks + x0
    basis = np.empty(n + 1, dtype=int)
    basis[:n] = m + 1 + np.arange(n)
    basis[-1] = 0

    # Canonicalize
    T, basis = build_phase2_tableau_canonical(A, b, c, basis)
    return T, basis


def change_to_zero_sum_phase2_only(GameMatrix):
    """
    Convert a zero-sum game matrix directly to a canonical Phase 2 tableau
    (same output shape as first_to_second would produce after removing AVs).
    Returns (T, basis, K).
    """
    m, n = GameMatrix.shape

    # Shift so B >= 0 and keep K for value unshift later
    min_element = np.min(GameMatrix)
    K = max(0.0, -min_element + 1e-6)
    B = GameMatrix + K

    # Standard-form A, b, c
    slack_matrix = np.eye(n)
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])
    A_sum         = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])
    A = np.vstack([A_constraints, A_sum])

    b = np.zeros(n + 1)
    b[-1] = 1.0

    c = np.zeros(m + 1 + n)
    c[m] = -1.0

    # Basis: all slacks + x0 (same reasoning as above)
    basis = np.empty(n + 1, dtype=int)
    basis[:n] = m + 1 + np.arange(n)
    basis[-1] = 0

    # Canonicalize
    T, basis = build_phase2_tableau_canonical(A, b, c, basis)
    return T, basis, K


if __name__ == '__main__':
    A = np.array([[-1, -1],
                  [-2, 3],
                  ])
    change_to_zero_sum(A)

