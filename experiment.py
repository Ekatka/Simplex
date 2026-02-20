import argparse
import numpy as np
import json
from collections import defaultdict

from stable_baselines3 import PPO
from scipy.stats import wilcoxon

from matrix import Matrix
from simplex_solver import (
    change_to_zero_sum_phase2_only,
    _pivot_col_heuristics, _pivot_row, _apply_pivot,
)
from envs import SecondPhasePivotingEnv
from config import (
    M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS,
    MODEL_NAME_TEMPLATE, PIVOT_MAP, PIVOT_MAP_TEST,
    PIVOT_STRATEGY_NAMES,
)
from base_matrix import BASE_MATRIX

MAXITER = 20_000
TOL = 1e-9


# ---------------------------------------------------------------------------
# Tableau preparation
# ---------------------------------------------------------------------------

def prepare_tableau(matrix_P):
    """Build Phase 2 tableau from a payoff matrix and remove artificial variables.

    Returns (T, basis) ready for pivoting, or None on failure.
    """
    res = change_to_zero_sum_phase2_only(matrix_P)
    if res is None:
        return None
    T, basis, K = res
    # Remove artificial variables from basis (mirrors SecondPhasePivotingEnv.remove_artificial)
    for pivrow in [row for row in range(basis.size) if basis[row] > T.shape[1] - 2]:
        non_zero_cols = [col for col in range(T.shape[1] - 1) if abs(T[pivrow, col]) > TOL]
        if non_zero_cols:
            _apply_pivot(T, basis, pivrow, non_zero_cols[0], TOL)
    return T, basis


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_fixed_strategy(T, basis, strategy):
    """Run a single fixed pivot strategy on a copy of the tableau.

    Returns dict with status, nit, objective.
    """
    T = T.copy()
    basis = basis.copy()
    nit = 0
    seen_bases = {tuple(int(i) for i in basis)}

    while nit < MAXITER:
        found, pivcol = _pivot_col_heuristics(T, strategy=strategy, tol=TOL)
        if not found:
            return {"status": "optimal", "nit": nit, "objective": float(T[-1, -1])}

        use_bland = (strategy == "blands_rule")
        found, pivrow = _pivot_row(T, basis, pivcol, phase=2, tol=TOL, bland=use_bland)
        if not found:
            return {"status": "no_pivot_row", "nit": nit, "objective": float(T[-1, -1])}

        _apply_pivot(T, basis, pivrow, pivcol, tol=TOL)
        nit += 1

        key = tuple(int(i) for i in basis)
        if key in seen_bases:
            return {"status": "loop", "nit": nit, "objective": float(T[-1, -1])}
        seen_bases.add(key)

    return {"status": "maxiter", "nit": nit, "objective": float(T[-1, -1])}


def run_rl_agent(T, basis, model):
    """Run the trained RL agent on a copy of the tableau.

    Returns dict with status, nit, objective.
    """
    env = SecondPhasePivotingEnv(T.copy(), basis.copy())
    obs, _ = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)

    status = info.get("status", "unknown")
    if truncated and not done:
        status = "loop"

    return {"status": status, "nit": env.nit, "objective": float(env.T[-1, -1])}


# ---------------------------------------------------------------------------
# Test matrix generation
# ---------------------------------------------------------------------------

def generate_test_matrices(n_matrices, mode="in_distribution"):
    """Generate a list of raw payoff matrices (np.ndarray).

    mode="in_distribution":  perturbations of the training base matrix.
    mode="out_of_distribution": fresh random matrices (new base each time).
    """
    matrices = []

    if mode == "in_distribution":
        base = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL,
                      epsilon=EPSILON, base_P=BASE_MATRIX)
        for _ in range(n_matrices):
            perturbed = base.generate_perturbed_matrix()
            matrices.append(perturbed.base_P)

    elif mode == "out_of_distribution":
        for _ in range(n_matrices):
            mat = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL, epsilon=0.0)
            mat.generateMatrix(mode="uniform")
            matrices.append(mat.base_P)

    return matrices


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(n_matrices, seed):
    np.random.seed(seed)

    model_path = MODEL_NAME_TEMPLATE.format(
        steps=TIMESTEPS, m=M, n=N,
        min=MIN_VAL, max=MAX_VAL, eps=EPSILON,
    )
    # SB3 appends .zip automatically, so strip it if present in template
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    strategies = list(PIVOT_MAP_TEST.values())
    all_methods = strategies + ["rl_agent"]

    results = {}
    for mode in ["in_distribution", "out_of_distribution"]:
        print(f"\n{'=' * 60}")
        print(f"  {mode.replace('_', ' ').upper()} ({n_matrices} matrices)")
        print(f"{'=' * 60}")

        matrices = generate_test_matrices(n_matrices, mode=mode)
        rows = []

        for i, matrix_P in enumerate(matrices):
            tableau = prepare_tableau(matrix_P)
            if tableau is None:
                print(f"  Matrix {i}: failed to build tableau, skipping")
                continue

            T, basis = tableau
            row = {"matrix_idx": i}

            for strategy in strategies:
                row[strategy] = run_fixed_strategy(T, basis, strategy)

            row["rl_agent"] = run_rl_agent(T, basis, model)
            rows.append(row)

            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{n_matrices}")

        results[mode] = rows
        print(f"  Done: {len(rows)} matrices solved")

    return results, all_methods


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(results, all_methods):
    strategies = [m for m in all_methods if m != "rl_agent"]

    for mode, rows in results.items():
        if not rows:
            continue

        n_total = len(rows)
        print(f"\n{'=' * 60}")
        print(f"  RESULTS: {mode.replace('_', ' ').upper()} ({n_total} matrices)")
        print(f"{'=' * 60}")

        # --- Convergence rates ---
        convergence = defaultdict(int)
        iters = defaultdict(list)
        for row in rows:
            for method in all_methods:
                if row[method]["status"] == "optimal":
                    convergence[method] += 1
                    iters[method].append(row[method]["nit"])

        print(f"\n  Convergence rates:")
        for method in all_methods:
            rate = convergence[method] / n_total * 100
            label = "RL Agent" if method == "rl_agent" else method
            print(f"    {label:25s}: {convergence[method]:4d}/{n_total} ({rate:.1f}%)")

        # --- Iteration statistics ---
        print(f"\n  Iteration counts (converged instances only):")
        print(f"    {'Method':25s} {'Mean':>8s} {'Median':>8s} {'Min':>6s} {'Max':>6s} {'N':>5s}")
        for method in all_methods:
            if iters[method]:
                arr = np.array(iters[method])
                label = "RL Agent" if method == "rl_agent" else method
                print(f"    {label:25s} {arr.mean():8.1f} {np.median(arr):8.1f} "
                      f"{arr.min():6d} {arr.max():6d} {len(arr):5d}")

        # --- Head-to-head: RL vs each heuristic ---
        print(f"\n  Head-to-head: RL Agent vs each heuristic")
        print(f"    {'Heuristic':25s} {'RL wins':>8s} {'Ties':>8s} {'RL loses':>9s} {'N/A':>5s}")

        for strategy in strategies:
            wins, ties, losses, na = 0, 0, 0, 0
            paired_rl, paired_heur = [], []

            for row in rows:
                rl_ok = row["rl_agent"]["status"] == "optimal"
                h_ok = row[strategy]["status"] == "optimal"

                if rl_ok and h_ok:
                    rl_n = row["rl_agent"]["nit"]
                    h_n = row[strategy]["nit"]
                    if rl_n < h_n:
                        wins += 1
                    elif rl_n == h_n:
                        ties += 1
                    else:
                        losses += 1
                    paired_rl.append(rl_n)
                    paired_heur.append(h_n)
                elif rl_ok and not h_ok:
                    wins += 1
                elif not rl_ok and h_ok:
                    losses += 1
                else:
                    na += 1

            print(f"    {strategy:25s} {wins:8d} {ties:8d} {losses:9d} {na:5d}")

            # Wilcoxon signed-rank test on paired converged instances
            if len(paired_rl) >= 10:
                diffs = np.array(paired_rl) - np.array(paired_heur)
                nonzero = diffs[diffs != 0]
                if len(nonzero) >= 10:
                    median_diff = np.median(nonzero)
                    alt = "less" if median_diff < 0 else "greater"
                    stat, p = wilcoxon(nonzero, alternative=alt)
                    direction = "RL better" if median_diff < 0 else "Heuristic better"
                    print(f"      Wilcoxon p={p:.6f} ({direction})")

        # --- RL vs best-per-instance heuristic ---
        print(f"\n  RL Agent vs best heuristic per instance:")
        wins, ties, losses, na = 0, 0, 0, 0

        for row in rows:
            rl_ok = row["rl_agent"]["status"] == "optimal"
            best_nit = None
            for strategy in strategies:
                if row[strategy]["status"] == "optimal":
                    if best_nit is None or row[strategy]["nit"] < best_nit:
                        best_nit = row[strategy]["nit"]

            if rl_ok and best_nit is not None:
                rl_n = row["rl_agent"]["nit"]
                if rl_n < best_nit:
                    wins += 1
                elif rl_n == best_nit:
                    ties += 1
                else:
                    losses += 1
            elif rl_ok and best_nit is None:
                wins += 1
            elif not rl_ok and best_nit is not None:
                losses += 1
            else:
                na += 1

        total_decided = wins + ties + losses
        print(f"    Wins: {wins:4d}  Ties: {ties:4d}  Losses: {losses:4d}  N/A: {na:4d}")
        if total_decided > 0:
            print(f"    Win rate: {wins / total_decided * 100:.1f}%  "
                  f"Win+Tie rate: {(wins + ties) / total_decided * 100:.1f}%")

        # --- Game value consistency check ---
        n_inconsistent = 0
        for row in rows:
            objectives = []
            for method in all_methods:
                if row[method]["status"] == "optimal":
                    objectives.append(row[method]["objective"])
            if objectives and (max(objectives) - min(objectives)) > 1e-4:
                n_inconsistent += 1

        if n_inconsistent > 0:
            print(f"\n  WARNING: {n_inconsistent} matrices had inconsistent "
                  f"game values across methods (diff > 1e-4)")
        else:
            print(f"\n  Game value consistency: all methods agree within 1e-4")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment: RL agent vs fixed pivot heuristics")
    parser.add_argument("--n-matrices", type=int, default=500,
                        help="Number of test matrices per test set (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save raw results to JSON file")
    args = parser.parse_args()

    print(f"Matrix size: {M}x{N}")
    print(f"Test matrices per set: {args.n_matrices}")
    print(f"Seed: {args.seed}")
    print(f"RL training strategies: {list(PIVOT_MAP.values())}")
    print(f"All tested strategies:  {list(PIVOT_MAP_TEST.values())}")

    results, all_methods = run_experiment(args.n_matrices, args.seed)
    analyze_results(results, all_methods)

    if args.save:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.save, "w") as f:
            json.dump(results, f, default=convert, indent=2)
        print(f"\nRaw results saved to {args.save}")


if __name__ == "__main__":
    main()