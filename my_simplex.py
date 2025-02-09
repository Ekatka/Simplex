import numpy as np
import gym
from gym import spaces
from warnings import warn


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):
    """
    Update tableau with given pivot row and column.
    Copied from _linprog_simplex.py
    """
    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] /= pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] -= T[pivrow] * T[irow, pivcol]

    if np.isclose(pivval, tol, atol=0, rtol=1e4):
        message = (
            f"The pivot operation produces a pivot value of:{pivval: .1e}, "
            "which is only slightly greater than the specified "
            f"tolerance{tol: .1e}. This may lead to issues regarding the "
            "numerical stability of the simplex method. "
            "Removing redundant constraints, changing the pivot strategy "
            "via Bland's rule or increasing the tolerance may "
            "help reduce the issue.")
        warn(message, UserWarning)


def _pivot_row_candidate(T, pivcol, phase, tol=1e-9):
    """
    Find pivoting row using min-ratio test
    Returns (row_found, row_idx).
    If row_found=False => unbounded.
    """
    if phase == 1:
        k = 2
    else:
        k = 1

    col_vals = T[:-k, pivcol]
    rhs_vals = T[:-k, -1]

    # We only consider rows where col_vals > tol
    ma = np.ma.masked_where(col_vals <= tol, col_vals, copy=False)
    if ma.count() == 0:
        # unbounded
        return False, np.nan

    mb = np.ma.masked_where(col_vals <= tol, rhs_vals, copy=False)
    ratio = mb / ma
    min_ratio = ratio.min()
    min_rows = np.ma.nonzero(ratio == min_ratio)[0]
    # pick the first for tie-break
    pivrow = min_rows[0]
    return True, pivrow


def _current_solution_vector(T, basis):
    """
    Reconstruct the current solution x from the tableau T.
    basis tells which columns are basic. T[:n_constraints, -1] are the basic variable values.
    used for steepest edge
    """
    num_constr = len(basis)
    dim = max(basis) + 1  # enough to hold the largest index in basis
    x = np.zeros(dim, dtype=np.float64)
    x[basis] = T[:num_constr, -1]
    return x


def _new_solution_if_pivot(T, basis, pivrow, pivcol):
    """
    Return the hypothetical new solution vector if we pivot on (pivrow, pivcol),
    without permanently modifying T.
    """
    T_copy = T.copy()
    basis_copy = basis.copy()
    _apply_pivot(T_copy, basis_copy, pivrow, pivcol)
    return _current_solution_vector(T_copy, basis_copy)


def _pivot_col_extended(T, basis, phase, tol=1e-9, pivot_method='bland', c=None):
    """
    Determine select pivot according to pivot_method:
      - 'bland'
      - 'largest_coefficient'
      - 'largest_increase'
      - 'steepest_edge'

    If no negative entries found, return (False, np.nan) => optimal.

    Parameters
    ----------
    T : 2D ndarray
        The simplex tableau.
    basis : 1D ndarray
        Current basis indices.
    phase : int
        1 or 2, used in row selection logic.
    tol : float
        Tolerance for negative entries in objective row.
    pivot_method : str
        Which pivot rule to apply.
    c : 1D ndarray or None
        The cost vector for 'steepest_edge'. If None and pivot_method='steepest_edge',
        we fallback to largest_coefficient or raise an error.

    Returns
    -------
    pivot_found : bool
        True if a pivot column is identified; False => no pivot col => optimal.
    pivcol : int or np.nan
        The chosen pivot column or NaN if none found.
    """
    # The last row is the objective row. Exclude the RHS from the search.
    obj_row = T[-1, :-1]
    # Mask out non-negative (>= -tol) => no improvement if we pivot there
    ma = np.ma.masked_where(obj_row >= -tol, obj_row, copy=False)
    if ma.count() == 0:
        # no negative entries => optimal
        return False, np.nan

    if pivot_method == 'bland':
        # pick the first negative index
        col_candidates = np.nonzero(~ma.mask)[0]
        return True, col_candidates[0]

    elif pivot_method == 'largest_coefficient':
        # pick the most negative
        return True, np.ma.nonzero(ma == ma.min())[0][0]

    elif pivot_method == 'largest_increase':
        # For each candidate col, we do a hypothetical pivot to see the improvement in objective
        best_improvement = None
        best_col = None
        col_candidates = np.nonzero(~ma.mask)[0]
        for j in col_candidates:
            row_found, i = _pivot_row_candidate(T, j, phase, tol=tol)

            if not row_found:
                # unbounded for this col => skip
                continue
            # The ratio is T[i,-1]/T[i,j], improvement in objective => ratio*T[-1,j]
            ratio = T[i, -1] / T[i, j]
            delta_obj = ratio * T[-1, j]
            improvement = abs(delta_obj)
            if (best_improvement is None) or (improvement > best_improvement):
                best_improvement = improvement
                best_col = j
        if best_col is None:
            # no valid pivot col
            return False, np.nan
        return True, best_col

    elif pivot_method == 'steepest_edge':
        # Need the cost vector 'c'. If not provided, fallback or raise.
        if c is None:
            warn("No cost vector 'c' provided for steepest_edge. Falling back to largest_coefficient.")
            return _pivot_col_extended(T, basis, phase, tol, 'largest_coefficient', None)

        best_score = None
        best_col = None
        col_candidates = np.nonzero(~ma.mask)[0]
        x_old = _current_solution_vector(T, basis)
        norm_c = np.linalg.norm(c)

        for j in col_candidates:
            row_found, i = _pivot_row_candidate(T, j, phase, tol=tol)

            if not row_found:
                continue
            x_new = _new_solution_if_pivot(T, basis, i, j)
            direction = x_new - x_old
            norm_direction = np.linalg.norm(direction)
            if norm_direction < 1e-14:
                # avoid division by zero
                continue

            numerator = np.dot(c, direction)
            denom = norm_c * norm_direction
            score = numerator / denom
            if (best_score is None) or (score > best_score):
                best_score = score
                best_col = j

        if best_col is None:
            return False, np.nan
        return True, best_col

    else:
        raise ValueError(f"Unknown pivot_method {pivot_method}")


def simplex_iteration_one_step(T, basis, phase, c, pivot_method='bland', tol=1e-9):
    """
    Perform one pivot iteration using the chosen pivot_method among:
    ['bland', 'largest_coefficient', 'largest_increase', 'steepest_edge'].

    Returns
    -------
    T : 2D ndarray
        Updated tableau.
    basis : 1D ndarray
        Updated basis if pivot occurred.
    status : int
        0 => pivot found and done step, or no pivot col => optimal,
        3 => unbounded.
    done : bool
        Whether we are done this LP (optimal or unbounded).
    message : str
        Explanation of the step result.
    """
    # find pivot column using _pivot_col_extended
    found_col, pivcol = _pivot_col_extended(T, basis, phase, tol, pivot_method, c=c)
    if not found_col:
        # no negative => optimal
        return T, basis, 0, True, "Optimal or no improvement"

    # find pivot row
    row_found, pivrow =_pivot_row_candidate(T, pivcol, phase, tol=tol)
    if not row_found:
        # unbounded
        return T, basis, 3, True, "Unbounded"

    # apply pivot
    _apply_pivot(T, basis, pivrow, pivcol, tol=tol)
    return T, basis, 0, False, "Pivot done"


class SimplexPivotEnv(gym.Env):
    """
    A Gym environment that solves a linear program in standard form:

        minimize c^T x
        subject to A x = b, x >= 0

    using a two-phase approach with artificial variables. We do:
      - Phase 1 to find a feasible solution, if possible.
      - Then transition to Phase 2 to optimize the real objective.

    At each step, the agent picks among:
      0 => 'bland'
      1 => 'largest_coefficient'
      2 => 'largest_increase'
      3 => 'steepest_edge'

    After each step give reward of -1
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, A, b, c, maxiter=1000, tol=1e-9):
        super(SimplexPivotEnv, self).__init__()
        self.A = A.copy()
        self.b = b.copy()
        self.c = c.copy()
        self.maxiter = maxiter
        self.tol = tol

        # Internal solver state
        self.T = None
        self.basis = None
        self.phase = 1
        self.av = None  # indices of artificial vars
        self.n = 0
        self.m = 0
        self.nit = 0
        self.done_flag = False
        self.status = 0
        self._build_phase1_tableau()

        self.action_space = spaces.Discrete(4)

        self.observation_space = None

    def _build_phase1_tableau(self):
        """
        Build the Phase 1 tableau
        """
        A = self.A
        b = self.b

        n, m = A.shape
        # ensure b >= 0
        is_neg = b < 0
        A[is_neg] *= -1
        b[is_neg] *= -1

        av = np.arange(n) + m  # artificial var indices
        self.basis = av.copy()

        row_constraints = np.hstack([A, np.eye(n), b.reshape(-1,1)])  # shape (n, m+n+1)
        row_objective = np.hstack([self.c, np.zeros(n), 0.0])         # shape (m+n+1)
        row_pseudo_objective = -row_constraints.sum(axis=0)           # shape (m+n+1)
        row_pseudo_objective[av] = 0.0

        T = np.vstack([row_constraints, row_objective, row_pseudo_objective])
        self.T = T
        self.n = n
        self.m = m
        self.av = av
        self.phase = 1
        self.nit = 0
        self.done_flag = False
        self.status = 0

    def reset(self):
        """
        Reset the environment: build a fresh Phase 1 tableau.
        Return the initial observation (flattened tableau).
        """
        self._build_phase1_tableau()

        # Define observation space shape
        flat_size = self.T.size
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(flat_size,), dtype=np.float32
        )
        return self._get_obs()

    def _get_obs(self):
        return self.T.flatten().astype(np.float32)

    def step(self, action):
        """
        action in {0,1,2,3} => which pivot strategy to use:
          0 => 'bland'
          1 => 'largest_coefficient'
          2 => 'largest_increase'
          3 => 'steepest_edge'
        Each step does exactly one pivot iteration.
        We also add reward = -1 each step to encourage fewer pivots.
        Large negative penalty if unbounded.
        Episode ends if no pivot col => optimal or if unbounded.
        If we finish Phase 1, we transition to Phase 2 automatically.
        """
        if self.done_flag:
            # If we're done, do nothing
            return self._get_obs(), 0.0, True, {}

        pivot_map = {
            0: 'bland',
            1: 'largest_coefficient',
            2: 'largest_increase',
            3: 'steepest_edge'
        }
        pivot_method = pivot_map.get(action, 'bland')
        print("Chosen pivoting method:", pivot_method)

        # one pivot iteration
        T, basis, step_status, done_step, msg = simplex_iteration_one_step(
            self.T, self.basis, self.phase, self.c,
            pivot_method=pivot_method,
            tol=self.tol
        )
        self.T = T
        self.basis = basis
        self.nit += 1

        # Base reward: -1 for each step
        reward = -1.0

        if step_status == 3:
            # unbounded
            self.status = 3
            self.done_flag = True
        elif done_step:
            # optimal
            self.status = step_status
            self.done_flag = True

        # If not done, handle Phase 1 -> Phase 2 transition
        if (not self.done_flag) and (self.phase == 1):
            # Check if pseudo-objective is near 0
            pseudo_obj_val = abs(self.T[-1, -1])
            if pseudo_obj_val < self.tol:
                # success => remove the last row (pseudo-objective)
                self.T = self.T[:-1, :]
                # remove artificial values
                self.T = np.delete(self.T, self.av, axis=1)
                self.phase = 2
            # else keep pivoting in Phase 1 until we find feasible or done

        # Check iteration limit
        if self.nit >= self.maxiter and not self.done_flag:
            self.status = 1
            self.done_flag = True

        obs = self._get_obs()
        info = {
            'status': self.status,
            'phase': self.phase,
            'nit': self.nit,
            'message': msg
        }
        return obs, reward, self.done_flag, info

    def render(self, mode='human'):
        print("Current Tableau:\n", self.T)
        print("Basis:", self.basis)
        print("Phase:", self.phase)
        print("Iteration count:", self.nit)
        print("Status code:", self.status)
        if self.phase == 2 or (self.phase == 1 and abs(self.T[-1, -1]) < self.tol):
            # if feasible
            x = self._current_solution()
            print("Current feasible solution x:", x)
            print("Objective value ~", float(self.T[-1, -1]))

    def _current_solution(self):
        """
        extracts values of basis column
        """

        x_full = np.zeros(self.m + self.n)  # up to m + n artificial
        x_full[self.basis[:self.n]] = self.T[:self.n, -1]
        return x_full[:self.m]  # original variables
