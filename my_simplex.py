import numpy as np
import gym
from gym import spaces
from warnings import warn

def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):
    """
    Update tableau with given pivot row and column.
    """
    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] /= pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] -= T[pivrow] * T[irow, pivcol]
    if np.isclose(pivval, tol, atol=0, rtol=1e4):
        message = (
            f"The pivot operation produces a pivot value of: {pivval:.1e}, "
            f"near the tolerance {tol:.1e}. Numerical issues may occur. "
            "Consider removing redundant constraints, changing pivot strategy, "
            "or increasing the tolerance."
        )
        warn(message, UserWarning)

def _pivot_row_candidate(T, pivcol, phase, tol=1e-9):
    """
    Find pivoting row using min-ratio test.
    Returns (row_found, row_idx).
    If row_found=False => unbounded.
    """
    # For phase 1, ignore last two rows; for phase 2, ignore last row.
    k = 2 if phase == 1 else 1

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

def _current_solution_vector(T, basis, fixed_size=None):
    """
    Reconstruct the current solution x from the tableau T.
    basis tells which columns are basic. T[:n_constraints, -1] are the basic variable values.
    If fixed_size is set, pad or truncate to that size.
    """
    num_constr = len(basis)
    dim = max(basis) + 1 if len(basis) > 0 else T.shape[1] - 1
    if fixed_size is not None:
        dim = max(dim, fixed_size)

    x = np.zeros(dim, dtype=np.float64)
    x[basis] = T[:num_constr, -1]  # Fill in basic variables
    return x[:fixed_size] if fixed_size else x

def _new_solution_if_pivot(T, basis, pivrow, pivcol, fixed_size=None):
    """
    Return the hypothetical new solution vector if we pivot on (pivrow, pivcol),
    without permanently modifying T.
    """
    T_copy = T.copy()
    basis_copy = basis.copy()
    _apply_pivot(T_copy, basis_copy, pivrow, pivcol)
    return _current_solution_vector(T_copy, basis_copy, fixed_size=fixed_size)

def _pivot_col_extended(T, basis, phase, tol=1e-9, pivot_method='bland', c=None):
    """
    Determine which pivot column to use according to pivot_method.
    If no negative entry in the objective row => optimal.
    """
    # The last row is the objective row (or pseudo-objective).
    # Exclude the right-hand side column from pivot selection.
    obj_row = T[-1, :-1]
    ma = np.ma.masked_where(obj_row >= -tol, obj_row, copy=False)
    if ma.count() == 0:
        return False, np.nan  # no negative => optimal

    fixed_size = T.shape[1] - 1

    if pivot_method == 'bland':
        # pick the first negative index
        col_candidates = np.nonzero(~ma.mask)[0]
        return True, col_candidates[0]

    elif pivot_method == 'largest_coefficient':
        # pick the most negative
        return True, np.ma.nonzero(ma == ma.min())[0][0]

    elif pivot_method == 'largest_increase':
        # Try each candidate col, see which yields largest objective improvement
        best_improvement = None
        best_col = None
        col_candidates = np.nonzero(~ma.mask)[0]
        for j in col_candidates:
            row_found, i = _pivot_row_candidate(T, j, phase, tol=tol)
            if not row_found:
                # unbounded for this col => skip
                continue
            ratio = T[i, -1] / T[i, j]  # pivot ratio
            delta_obj = ratio * T[-1, j]
            improvement = abs(delta_obj)
            if (best_improvement is None) or (improvement > best_improvement):
                best_improvement = improvement
                best_col = j
        if best_col is None:
            return False, np.nan
        return True, best_col

    elif pivot_method == 'steepest_edge':
        # If c is None, fallback
        if c is None:
            warn("No cost vector 'c' for steepest_edge. Fallback to largest_coefficient.")
            return _pivot_col_extended(T, basis, phase, tol, 'largest_coefficient')

        best_score = None
        best_col = None
        col_candidates = np.nonzero(~ma.mask)[0]
        x_old = _current_solution_vector(T, basis, fixed_size=fixed_size)
        norm_c = np.linalg.norm(c)

        for j in col_candidates:
            row_found, i = _pivot_row_candidate(T, j, phase, tol=tol)
            if not row_found:
                continue
            x_new = _new_solution_if_pivot(T, basis, i, j, fixed_size=fixed_size)
            direction = x_new - x_old
            norm_direction = np.linalg.norm(direction)
            if norm_direction < 1e-14:
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

    # Fallback
    return _pivot_col_extended(T, basis, phase, tol, 'bland')

def simplex_iteration_one_step(T, basis, phase, c, pivot_method='bland', tol=1e-9):
    """
    Perform one pivot iteration using the chosen method.
    Returns updated tableau T, updated basis, status code, done boolean, and message.
    """
    # 1) Choose pivot column
    found_col, pivcol = _pivot_col_extended(T, basis, phase, tol, pivot_method, c=c)
    if not found_col:
        # no negative => optimal
        return T, basis, 0, True, "Optimal or no improvement"

    # 2) Choose pivot row
    row_found, pivrow = _pivot_row_candidate(T, pivcol, phase, tol=tol)
    if not row_found:
        # unbounded
        return T, basis, 3, True, "Unbounded"

    # 3) Apply pivot
    _apply_pivot(T, basis, pivrow, pivcol, tol=tol)
    return T, basis, 0, False, "Pivot done"


class SimplexGymEnv(gym.Env):
    """
    A Gym environment that solves a linear program using a two-phase simplex approach.

    Actions: 0 => 'bland'
             1 => 'largest_coefficient'
             2 => 'largest_increase'
             3 => 'steepest_edge'
    Reward: -1 each step (encourage fewer pivots)
    Episode ends if: optimal or unbounded or iteration limit reached.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, A, b, c, maxiter=1000, tol=1e-9):
        super().__init__()
        self.A = A.copy()
        self.b = b.copy()
        self.c = c.copy()
        self.maxiter = maxiter
        self.tol = tol

        # Internal solver state
        self.T = None
        self.basis = None
        self.phase = 1
        self.av = None
        self.n = 0
        self.m = 0
        self.nit = 0
        self.done_flag = False
        self.status = 0

        # Build the initial Phase 1 tableau
        self._build_phase1_tableau()

        # Action space: 4 discrete pivot strategies
        self.action_space = spaces.Discrete(3)

        # Define a fixed-size observation space to avoid shape mismatch
        max_rows = self.A.shape[0] + 2        # original constraints + 2 extra rows (obj and pseudo-obj)
        max_cols = self.A.shape[1] + self.A.shape[0] + 1  # original vars + artificial + RHS
        self.max_obs_size = max_rows * max_cols
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(self.max_obs_size,), dtype=np.float32
        )

    def _build_phase1_tableau(self):
        """
        Construct the Phase 1 tableau with artificial variables.
        """
        A, b = self.A, self.b
        n, m = A.shape

        # Ensure b >= 0
        negative_rows = (b < 0)
        A[negative_rows] *= -1
        b[negative_rows] *= -1

        # Introduce artificial variables
        self.av = np.arange(n) + m
        self.basis = self.av.copy()

        # Build constraints rows
        row_constraints = np.hstack([A, np.eye(n), b.reshape(-1, 1)])  # shape: (n, m + n + 1)
        # Build real objective row
        row_objective = np.hstack([self.c, np.zeros(n), 0.0])          # shape: (m + n + 1)
        # Build pseudo-objective row
        row_pseudo_objective = -row_constraints.sum(axis=0)            # shape: (m + n + 1)
        row_pseudo_objective[self.av] = 0.0

        # Combine into one tableau
        self.T = np.vstack([row_constraints, row_objective, row_pseudo_objective])
        self.n = n
        self.m = m
        self.phase = 1
        self.nit = 0
        self.done_flag = False
        self.status = 0

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a fresh Phase 1 tableau.
        Returns: observation (flattened, padded), info dict
        """
        super().reset(seed=seed)
        self._build_phase1_tableau()
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Return a flattened, padded copy of the tableau so the obs space size is always consistent.
        """
        obs = self.T.flatten().astype(np.float32)
        padded_obs = np.zeros(self.max_obs_size, dtype=np.float32)
        padded_obs[:obs.size] = obs
        return padded_obs

    def step(self, action):
        """
        Perform one pivot iteration based on the chosen pivot strategy.
        Return: (obs, reward, done, truncated, info)
        """
        if self.done_flag:
            return self._get_obs(), 0.0, True, False, {}
        if isinstance(action, np.ndarray):
            action = int(action.item())

        pivot_map = {
            0: 'bland',
            1: 'largest_coefficient',
            2: 'largest_increase',
            # 3: 'steepest_edge'
        }
        pivot_method = pivot_map.get(action, 'bland')

        # Perform one simplex iteration
        self.T, self.basis, step_status, done_step, msg = simplex_iteration_one_step(
            self.T, self.basis, self.phase, self.c, pivot_method=pivot_method, tol=self.tol
        )
        self.nit += 1

        # Default reward is -1 per pivot
        reward = -1.0

        # Check status from pivot
        if step_status == 3:
            # Unbounded
            self.status = 3
            self.done_flag = True
        elif done_step:
            # Optimal
            self.status = step_status
            self.done_flag = True

        # Phase 1 -> Phase 2 transition if pseudo-objective is near 0
        if (not self.done_flag) and (self.phase == 1):
            pseudo_obj_val = abs(self.T[-1, -1])
            if pseudo_obj_val < self.tol:
                # Remove pseudo-objective row
                self.T = self.T[:-1, :]
                # Remove artificial columns
                self.T = np.delete(self.T, self.av, axis=1)
                self.phase = 2

        # Check iteration limit
        if self.nit >= self.maxiter and not self.done_flag:
            self.status = 1
            self.done_flag = True

        obs = self._get_obs()
        done = self.done_flag
        info = {
            'status': self.status,
            'phase': self.phase,
            'nit': self.nit,
            'message': msg
        }
        return obs, reward, done, False, info

    def render(self, mode='human'):
        """
        Print the current tableau and solver state.
        """
        print("Current Tableau:\n", self.T)
        print("Basis:", self.basis)
        print("Phase:", self.phase)
        print("Iteration count:", self.nit)
        print("Status code:", self.status)

        # Print current feasible solution if feasible
        if self.phase == 2 or (self.phase == 1 and abs(self.T[-1, -1]) < self.tol):
            x = self._current_solution()
            print("Current feasible solution x:", x)
            print("Objective value:", float(self.T[-1, -1]))

    def _current_solution(self):
        """
        Extract the current feasible solution from the tableau.
        The basis array tracks which columns are basic in the first 'n' rows.
        """
        x_full = np.zeros(self.m + self.n, dtype=np.float64)
        x_full[self.basis[:self.n]] = self.T[:self.n, -1]
        return x_full[:self.m]  # Return only original variable portion
