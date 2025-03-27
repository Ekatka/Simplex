import gym
import numpy as np
from warnings import warn
import scipy.sparse as sps
from collections import namedtuple
import gym
from gym import spaces

from _linprog_utils import (
    _parse_linprog, _presolve, _get_Abc, _LPProblem, _autoscale,
    _postsolve, _check_result, _display_summary)

class PivotingEnv(gym.Env):

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
        self.maxiter = 5000
        self.nit = 0
        if len(self.basis[:self.m]) == 0:
            self.solution = np.empty(self.T.shape[1] - 1, dtype=np.float64)
        else:
            self.solution = np.empty(max(self.T.shape[1] - 1, max(self.basis[:self.m]) + 1),
                                dtype=np.float64)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.T.shape, dtype=np.float64
        )
        self.complete = False;

    def _get_obs(self):
        return self.T.copy()

    def reset(self, seed=None, **kwargs):
        self.nit = 0
        return self._get_obs(), {}

    def step(self, action):
        strategy_map = {
            0: 'bland',
            1: 'largest_coefficient',
            2: 'largest_increase',
            3: 'steepest_edge'
        }
        strategy = strategy_map[int(action)]
        done = False
        reward = -1  # Penalize per step only

        pivcol_found, pivcol = _pivot_col_phase2(self.T, strategy=strategy,tol=self.tol )
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

def _pivot_col_phase2(T, strategy, tol=1e-9):
    """
    Pick pivot column using one of the supported heuristics.
    """
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan

    if strategy == 'bland':
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


def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):

    if phase == 1:
        k = 2
    else:
        k = 1
    ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
    if ma.count() == 0:
        return False, np.nan
    mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
    q = mb / ma
    min_rows = np.ma.nonzero(q == q.min())[0]
    if bland:
        return True, min_rows[np.argmin(np.take(basis, min_rows))]
    return True, min_rows[0]


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):

    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] = T[pivrow] / pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] = T[irow] - T[pivrow] * T[irow, pivcol]

    # The selected pivot should never lead to a pivot value less than the tol.
    if np.isclose(pivval, tol, atol=0, rtol=1e4):
        message = (
            f"The pivot operation produces a pivot value of:{pivval: .1e}, "
            "which is only slightly greater than the specified "
            f"tolerance{tol: .1e}. This may lead to issues regarding the "
            "numerical stability of the simplex method. "
            "Removing redundant constraints, changing the pivot strategy "
            "via Bland's rule or increasing the tolerance may "
            "help reduce the issue.")
        warn(message, stacklevel=5)

def phase1solver(T, n, basis,
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


def solve_zero_sum(GameMatrix):
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

    # As all constraints are equality constraints the artificial variables
    # will also be basic variables.
    av = np.arange(n) + m
    basis = av.copy()

    # Format the phase one tableau by adding artificial variables and stacking
    # the constraints, the objective row and pseudo-objective row.
    row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
    row_objective = np.hstack((c, np.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    nit1, status = phase1solver(T, n, basis)
    tol = tol=1e-9
    nit2 = nit1
    if abs(T[-1, -1]) < tol:
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
            f"which exceeds the required tolerance of {tol} for a solution to be "
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
        print(f"[solve_zero_sum] Pseudo-objective: {T[-1, -1]:.2e} vs tol {tol:.1e}, status={status}")
        return None


if __name__ == '__main__':
    A = np.array([[-1, -1],
                  [-2, 3],
                  ])
    solve_zero_sum(A)










