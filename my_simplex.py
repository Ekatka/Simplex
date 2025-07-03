import numpy as np
from warnings import warn



def _pivot_col(T, tol=1e-9, strategy='largest_coefficient'):
    """
    Pick pivot column using one of the supported heuristics.
    """
    cost_row = T[-1, :-1]

    if strategy == 'bland':
        for j, val in enumerate(cost_row):
            if val < -tol:
                return True, j
        return False, np.nan

    elif strategy == 'largest_coefficient':
        masked = np.ma.masked_where(cost_row >= -tol, cost_row)
        if masked.count() == 0:
            return False, np.nan
        return True, np.ma.nonzero(masked == masked.min())[0][0]

    elif strategy == 'largest_increase':
        # Delta_obj = abs(c_j) / (pivot column ratio) - requires extra work
        col_indices = [j for j in range(len(cost_row)) if cost_row[j] < -tol]
        best_increase = -np.inf
        best_col = None
        for j in col_indices:
            ratios = []
            for i in range(len(T) - 1):
                if T[i, j] > tol:
                    ratio = T[i, -1] / T[i, j]
                    increase = abs(cost_row[j]) * ratio
                    ratios.append((increase, j))
            if ratios:
                max_inc, col = max(ratios)
                if max_inc > best_increase:
                    best_increase = max_inc
                    best_col = col
        if best_col is None:
            return False, np.nan
        return True, best_col

    elif strategy == 'steepest_edge':
        best_ratio = np.inf
        best_col = None
        for j in range(len(cost_row)):
            if cost_row[j] < -tol:
                norm = np.linalg.norm(T[:-1, j])
                if norm == 0:
                    continue
                ratio = abs(cost_row[j]) / norm
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_col = j
        if best_col is None:
            return False, np.nan
        return True, best_col

    else:
        raise ValueError(f"Unknown pivot strategy: {strategy}")


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


def _solve_simplex(T, n, basis, callback, postsolve_args,
                   maxiter=1000, tol=1e-9, phase=2, bland=False, nit0=0,
                   ):

    nit = nit0
    status = 0
    message = ''
    complete = False

    if phase == 1:
        m = T.shape[1]-2
    elif phase == 2:
        m = T.shape[1]-1
    else:
        raise ValueError("Argument 'phase' to _solve_simplex must be 1 or 2")

    if phase == 2:
        # Check if any artificial variables are still in the basis.
        # If yes, check if any coefficients from this row and a column
        # corresponding to one of the non-artificial variable is non-zero.
        # If found, pivot at this term. If not, start phase 2.
        # Do this for all artificial variables in the basis.
        # Ref: "An Introduction to Linear Programming and Game Theory"
        # by Paul R. Thie, Gerard E. Keough, 3rd Ed,
        # Chapter 3.7 Redundant Systems (pag 102)
        for pivrow in [row for row in range(basis.size)
                       if basis[row] > T.shape[1] - 2]:
            non_zero_row = [col for col in range(T.shape[1] - 1)
                            if abs(T[pivrow, col]) > tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1

    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1),
                            dtype=np.float64)

    while not complete:
        # Find the pivot column
        pivcol_found, pivcol = _pivot_col(T, tol, bland)
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

        # if callback is not None:
        #     solution[:] = 0
        #     solution[basis[:n]] = T[:n, -1]
        #     x = solution[:m]
        #     x, fun, slack, con = _postsolve(
        #         x, postsolve_args
        #     )
        #     res = OptimizeResult({
        #         'x': x,
        #         'fun': fun,
        #         'slack': slack,
        #         'con': con,
        #         'status': status,
        #         'message': message,
        #         'nit': nit,
        #         'success': status == 0 and complete,
        #         'phase': phase,
        #         'complete': complete,
        #         })
        #     callback(res)

        if not complete:
            if nit >= maxiter:
                # Iteration limit exceeded
                status = 1
                complete = True
            else:
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    return nit, status


def _linprog_simplex(c, c0, A, b, callback, postsolve_args,
                     maxiter=1000, tol=1e-9, disp=False, bland=False,
                     **unknown_options):


    status = 0
    messages = {0: "Optimization terminated successfully.",
                1: "Iteration limit reached.",
                2: "Optimization failed. Unable to find a feasible"
                   " starting point.",
                3: "Optimization failed. The problem appears to be unbounded.",
                4: "Optimization failed. Singular matrix encountered."}

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

    nit1, status = _solve_simplex(T, n, basis, callback=callback,
                                  postsolve_args=postsolve_args,
                                  maxiter=maxiter, tol=tol, phase=1,
                                  bland=bland
                                  )
    # if pseudo objective is zero, remove the last row from the tableau and
    # proceed to phase 2
    nit2 = nit1
    if abs(T[-1, -1]) < tol:
        # Remove the pseudo-objective row from the tableau
        T = T[:-1, :]
        # Remove the artificial variable columns from the tableau
        T = np.delete(T, av, 1)
    else:
        # Failure to find a feasible starting point
        status = 2
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
        nit2, status = _solve_simplex(T, n, basis, callback=callback,
                                      postsolve_args=postsolve_args,
                                      maxiter=maxiter, tol=tol, phase=2,
                                      bland=bland, nit0=nit1
                                      )

    solution = np.zeros(n + m)
    solution[basis[:n]] = T[:n, -1]
    x = solution[:m]

    return x, status, messages[status], int(nit2)

def prepare_phase2_tableau(A, b, c, tol=1e-9, maxiter=1000):
    """
    Solves Phase 1 using Bland's rule to obtain a feasible starting point.
    Returns Phase 2 tableau and feasible basis.
    """
    m, n = A.shape

    # Step 1: Create Phase 1 tableau
    T = np.hstack([A, np.eye(m), b.reshape(-1, 1)])  # Add artificial vars and RHS
    cost_row = np.hstack([np.zeros(n), np.ones(m), 0])
    T = np.vstack([T, cost_row])

    basis = np.arange(n, n + m)

    # Step 2: Perform simplex iterations (Phase 1)
    for _ in range(maxiter):
        found, pivcol = _pivot_col(T, tol=tol, strategy='bland')
        if not found:
            break

        found, pivrow = _pivot_row(T, basis, pivcol, phase=1, tol=tol)
        if not found:
            raise ValueError("Phase 1 detected problem is infeasible.")

        _apply_pivot(T, basis, pivrow, pivcol, tol=tol)

    # Step 3: Check feasibility
    if abs(T[-1, -1]) > tol:
        raise ValueError(f"Problem is infeasible. Phase 1 objective: {T[-1, -1]}")

    # Step 4: Build Phase 2 tableau
    T = T[:-1, :]  # Remove last row (phase 1 objective)

    # Track which columns will remain (non-artificial + RHS)
    keep_cols = np.r_[np.arange(n), [T.shape[1] - 1]]  # original vars + RHS
    T = T[:, keep_cols]  # Remove artificial variable columns

    # Create correct-sized Phase 2 objective row
    num_cols = T.shape[1]
    padding_len = num_cols - len(c) - 1
    if padding_len < 0:
        raise ValueError(f"Too many objective coefficients: len(c) = {len(c)}, but tableau has only {num_cols} columns")

    phase2_obj = np.hstack([c, np.zeros(padding_len), 0])
    T = np.vstack([T, phase2_obj])

    # Step 5: Fix objective row using updated basis
    col_map = keep_cols  # original column indices that were kept
    new_basis = []
    for i, var in enumerate(basis):
        if var in col_map:
            new_var = np.where(col_map == var)[0][0]
            T[-1] -= T[-1, new_var] * T[i]
            new_basis.append(new_var)

    basis = np.array(new_basis, dtype=int)

    # Step 6: Recover feasible basis if lost
    if len(basis) == 0:
        print("[Warning] Empty basis after Phase 1 — attempting recovery from tableau.")
        basis = []
        num_rows = T.shape[0] - 1
        num_cols = T.shape[1]
        for col in range(num_cols - 1):  # exclude RHS
            column = T[:num_rows, col]
            if np.count_nonzero(column) == 1 and np.isclose(column.max(), 1):
                row = np.argmax(column)
                if all(np.isclose(T[:num_rows, col], [1 if i == row else 0 for i in range(num_rows)])):
                    basis.append(col)
        basis = np.array(basis, dtype=int)

    return T, basis

