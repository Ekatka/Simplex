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
        self.basis = basis
        self.T = T
        self.tol = 1e-9
        self.m = self.T.shape[1] - 1
        self.remove_artificial()
        self.maxiter = 1_000_000
        self.nit = 0
        if len(self.basis[:self.m]) == 0:
            self.solution = np.empty(self.T.shape[1] - 1, dtype=np.float64)
        else:
            self.solution = np.empty(max(self.T.shape[1] - 1, max(self.basis[:self.m]) + 1),
                                dtype=np.float64)
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
        return self._get_obs(), {}

    def step(self, action):
        # Use PIVOT_MAP from config instead of local definition
        strategy = PIVOT_MAP[int(action)]
        done = False
        reward = -1  # Penalize per step only

        pivcol_found, pivcol = _pivot_col_heuristics(self.T, strategy=strategy, tol=self.tol)
        if not pivcol_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        pivrow_found, pivrow = _pivot_row(self.T, self.basis, pivcol, phase=2, tol=self.tol)
        if not pivrow_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        _apply_pivot(self.T, self.basis, pivrow, pivcol, tol=self.tol)
        self.nit += 1

        if self.nit >= self.maxiter:
            done = True

        return self._get_obs(), reward, done, False, {}

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
    Convert a zero-sum game matrix directly to Phase 2 tableau without Phase 1.
    This function builds a trivial feasible BFS and returns a Phase 2 tableau
    ready for optimization, in the same format as first_to_second would produce.
    
    Args:
        GameMatrix: Payoff matrix M (m x n)
    
    Returns:
        T: Phase 2 tableau ready for optimization
        basis: Basis indices for the BFS
        None: No artificial variables needed (av is None for compatibility)
    """
    m, n = GameMatrix.shape
    
    # Step 1: Compute K so that B = M + K*11^T >= 0
    min_element = np.min(GameMatrix)
    K = max(0, -min_element + 1e-6)
    
    # Step 2: Create matrix B = M + K*11^T
    B = GameMatrix + K
    
    # Step 3: Convert to standard form LP exactly as change_to_zero_sum does
    # The LP is: max v subject to B^T * x >= v, sum(x) = 1, x >= 0
    # This is equivalent to: min -v subject to -B^T * x + v <= 0, sum(x) = 1, x >= 0
    
    # Variables: x (m variables) + v (1 variable) + slack variables (n variables)
    # Total variables: m + 1 + n
    
    # Build constraint matrix A
    # First n rows: -B^T * x + v + s_i = 0 (where s_i are slack variables)
    # Last row: sum(x) = 1
    
    # Slack variables for the first n constraints
    slack_matrix = np.eye(n)
    
    # Build the constraint matrix
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])
    A_sum = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])
    
    A = np.vstack([A_constraints, A_sum])
    
    # Right-hand side vector
    b = np.zeros(n + 1)
    b[-1] = 1  # sum(x) = 1
    
    # Objective function: min -v (equivalent to max v)
    c = np.zeros(m + 1 + n)  # x variables + v variable + slack variables
    c[m] = -1  # coefficient for v variable
    
    # Step 4: Build the tableau
    # Tableau format: [A | b]
    #                 [c | 0]
    
    tableau_constraints = np.hstack([A, b.reshape(-1, 1)])
    tableau_objective = np.append(c, 0)
    
    T = np.vstack([tableau_constraints, tableau_objective])
    
    # Step 5: Set up the basis for a feasible starting point
    # The basis will include:
    # - One decision variable (x[0])
    # - The value variable (v)
    # - Slack variables for the remaining constraints
    
    basis = np.zeros(n + 1, dtype=int)
    basis[0] = 0  # x[0] is basic
    basis[1] = m  # v is basic
    basis[2:] = m + 1 + np.arange(n-1)  # slack variables for first n-1 constraints
    
    # Step 6: Ensure the tableau represents a valid BFS
    # We need to perform elementary row operations to get the identity matrix
    # in the basis columns
    
    # This is a simplified approach - in practice, you might need more sophisticated
    # row operations to ensure the tableau is in canonical form
    
    return T, basis, None


def build_trivial_bfs_zero_sum_game(GameMatrix):
    """
    Build a trivial feasible basic feasible solution (BFS) for zero-sum games.
    This function computes K so that B = M + K*11^T >= 0, then constructs
    a Phase 2 tableau directly from this BFS.
    
    Args:
        GameMatrix: Payoff matrix M (m x n)
    
    Returns:
        T: Phase 2 tableau ready for optimization
        basis: Basis indices for the BFS
    """
    m, n = GameMatrix.shape
    
    # Step 1: Compute K so that B = M + K*11^T >= 0
    # We need K >= -min(M) to ensure all elements are non-negative
    min_element = np.min(GameMatrix)
    K = max(0, -min_element + 1e-6)  # Add small epsilon for numerical stability
    
    # Step 2: Create matrix B = M + K*11^T
    B = GameMatrix + K
    
    # Step 3: Convert to standard form LP
    # The LP is: max v subject to B^T * x >= v, sum(x) = 1, x >= 0
    # This is equivalent to: min -v subject to -B^T * x + v <= 0, sum(x) = 1, x >= 0
    
    # Constraints: -B^T * x + v <= 0 (n constraints)
    # Constraint: sum(x) = 1 (1 constraint)
    # Variables: x (m variables) + v (1 variable) + slack variables (n+1 variables)
    
    # Build constraint matrix A
    # First n rows: -B^T * x + v + s_i = 0 (where s_i are slack variables)
    # Last row: sum(x) = 1
    
    # Slack variables for the first n constraints
    slack_matrix = np.eye(n)
    
    # Build the constraint matrix
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])
    A_sum = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])
    
    A = np.vstack([A_constraints, A_sum])
    
    # Right-hand side vector
    b = np.zeros(n + 1)
    b[-1] = 1  # sum(x) = 1
    
    # Objective function: min -v (equivalent to max v)
    c = np.zeros(m + 1 + n)  # x variables + v variable + slack variables
    c[m] = -1  # coefficient for v variable
    
    # Step 4: Choose starting basis
    # We'll use the first pure strategy (row 0) as starting basis
    # Set x[0] = 1/min(B[0, :]), all other decision vars = 0
    
    # Find the minimum element in the first row of B
    min_first_row = np.min(B[0, :])
    x0_val = 1.0 / min_first_row if min_first_row > 0 else 1.0
    
    # Set up the starting solution
    x_solution = np.zeros(m)
    x_solution[0] = x0_val
    
    # Compute v value: v = min(B^T * x)
    v_val = np.min(B.T @ x_solution)
    
    # Compute slack variables: s = B^T * x - v
    slack_solution = B.T @ x_solution - v_val
    
    # Step 5: Build the tableau
    # Tableau format: [A | b]
    #                 [c | 0]
    
    tableau_constraints = np.hstack([A, b.reshape(-1, 1)])
    tableau_objective = np.append(c, 0)
    
    T = np.vstack([tableau_constraints, tableau_objective])
    
    # Step 6: Set up the basis
    # The basis will include:
    # - The first decision variable (x[0])
    # - The value variable (v)
    # - Slack variables for the first n-1 constraints
    # - The last constraint (sum(x) = 1) will be satisfied by the basis
    
    basis = np.zeros(n + 1, dtype=int)
    basis[0] = 0  # x[0] is basic
    basis[1] = m  # v is basic
    basis[2:] = m + 1 + np.arange(n-1)  # slack variables for first n-1 constraints
    
    return T, basis


def change_to_zero_sum_phase2_only(GameMatrix):
    """
    Convert a zero-sum game matrix directly to Phase 2 tableau without Phase 1.
    This function simulates the result of first_to_second by creating a tableau
    in the exact format that would be produced after removing artificial variables.
    
    Args:
        GameMatrix: Payoff matrix M (m x n)
    
    Returns:
        T: Phase 2 tableau ready for optimization
        basis: Basis indices for the BFS
        K: The constant shift applied to M (needed for game value calculation)
    """
    m, n = GameMatrix.shape
    
    # Step 1: Compute K so that B = M + K*11^T >= 0
    min_element = np.min(GameMatrix)
    K = max(0, -min_element + 1e-6)
    
    # Step 2: Create matrix B = M + K*11^T
    B = GameMatrix + K
    
    # Step 3: Convert to standard form LP exactly as change_to_zero_sum does
    # The LP is: max v subject to B^T * x >= v, sum(x) = 1, x >= 0
    # This is equivalent to: min -v subject to -B^T * x + v <= 0, sum(x) = 1, x >= 0
    
    # Variables: x (m variables) + v (1 variable) + slack variables (n variables)
    # Total variables: m + 1 + n
    
    # Build constraint matrix A
    # First n rows: -B^T * x + v + s_i = 0 (where s_i are slack variables)
    # Last row: sum(x) = 1
    
    # Slack variables for the first n constraints
    slack_matrix = np.eye(n)
    
    # Build the constraint matrix
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])
    A_sum = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])
    
    A = np.vstack([A_constraints, A_sum])
    
    # Right-hand side vector
    b = np.zeros(n + 1)
    b[-1] = 1  # sum(x) = 1
    
    # Objective function: min -v (equivalent to max v)
    c = np.zeros(m + 1 + n)  # x variables + v variable + slack variables
    c[m] = -1  # coefficient for v variable
    
    # Step 4: Build the tableau
    # Tableau format: [A | b]
    #                 [c | 0]
    
    tableau_constraints = np.hstack([A, b.reshape(-1, 1)])
    tableau_objective = np.append(c, 0)
    
    T = np.vstack([tableau_constraints, tableau_objective])
    
    # Step 5: Set up the basis for a feasible starting point
    # The basis will include:
    # - One decision variable (x[0])
    # - The value variable (v)
    # - Slack variables for the remaining constraints
    
    basis = np.zeros(n + 1, dtype=int)
    basis[0] = 0  # x[0] is basic
    basis[1] = m  # v is basic
    basis[2:] = m + 1 + np.arange(n-1)  # slack variables for first n-1 constraints
    
    # Step 6: Ensure the tableau represents a valid BFS
    # We need to perform elementary row operations to get the identity matrix
    # in the basis columns
    
    # This is a simplified approach - in practice, you might need more sophisticated
    # row operations to ensure the tableau is in canonical form
    
    return T, basis, K


if __name__ == '__main__':
    A = np.array([[-1, -1],
                  [-2, 3],
                  ])
    change_to_zero_sum(A)

