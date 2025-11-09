import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import itertools
import json
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs import RandomMatrixEnv
from matrix import Matrix
from config import MATRIX_MODE, TOEPLITZ_RHO, TOEPLITZ_SIGNED, TOEPLITZ_ANTISYMMETRIC, TOEPLITZ_BAND
from config import M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS, N_ENVS, MODEL_NAME_TEMPLATE, LOAD_MODEL, PREFERRED_ACTION_ID, INITIAL_BIAS, USE_MACRO_STRATEGY, USE_BIAS_ANNEALING, USE_INITIAL_ACTION_BIAS, USE_HISTORY_TRACKING, HISTORY_SIZE, NO_IMPROVE_STEPS
from base_matrix import BASE_MATRIX

from wrappers import MacroStrategyWrapper
from callbacks import EpisodeCounterCallback, LogitBiasAnnealCallback, HistoryTrackerCallback
from io_utils import update_base_matrix


def apply_initial_action_bias(model, preferred_id: int, initial_bias: float = 3.0):
    import torch as th
    with th.no_grad():
        bias = model.policy.action_net.bias
        bias[:] = 0.0
        bias[int(preferred_id)] = float(initial_bias)


def create_ppo_model(vec_env, verbose=1, n_envs=1, learning_rate=3e-5, n_steps=1024, clip_range=0.2):
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    return PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=verbose,
        gamma=0.995,
        n_steps=n_steps,
        batch_size=max(64, 2048//max(1, n_envs)),
        ent_coef=0.01,
        learning_rate=learning_rate,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs
    )


def train_single_config(matrix, learning_rate, n_steps, clip_range, run_id, total_runs):
    """Trains a model with given hyperparameters"""
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"Grid Search: Run {run_id}/{total_runs}")
    print(f"Hyperparameters: lr={learning_rate:.2e}, n_steps={n_steps}, clip_range={clip_range}")
    print(f"{'='*80}\n")

    def make_env():
        base_env = RandomMatrixEnv(matrix)
        if USE_MACRO_STRATEGY:
            base_env = MacroStrategyWrapper(base_env, macro_len=10)
        return TimeLimit(base_env, max_episode_steps=2000)

    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    model = create_ppo_model(
        vec_env, 
        n_envs=N_ENVS,
        learning_rate=learning_rate,
        n_steps=n_steps,
        clip_range=clip_range
    )

    if USE_INITIAL_ACTION_BIAS:
        apply_initial_action_bias(model, PREFERRED_ACTION_ID, INITIAL_BIAS)

    callbacks = [EpisodeCounterCallback()]

    if USE_BIAS_ANNEALING:
        bias_callback = LogitBiasAnnealCallback(
            preferred_id=PREFERRED_ACTION_ID,
            initial_bias=INITIAL_BIAS,
            half_life=200_000
        )
        callbacks.append(bias_callback)

    if USE_HISTORY_TRACKING:
        history_callback = HistoryTrackerCallback(
            history_size=HISTORY_SIZE,
            no_improve_steps=NO_IMPROVE_STEPS
        )
        callbacks.append(history_callback)

    # Train the model
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)

    # Save the model with a unique name
    # Format learning_rate for filename (replace dots and signs with underscores)
    lr_str = f"{learning_rate:.2e}".replace(".", "p").replace("-", "m").replace("+", "p")
    filename = f"models/ppo_grid_lr{lr_str}_nsteps{n_steps}_clip{clip_range}_matrix{M}x{N}_min{MIN_VAL}_max{MAX_VAL}_epsilon{EPSILON}.zip"
    model.save(filename)

    elapsed_time = time.time() - start_time

    result = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "clip_range": clip_range,
        "model_path": filename,
        "timesteps": TIMESTEPS,
        "training_time_seconds": elapsed_time
    }

    print(f"Model saved: {filename}")
    print(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    return result


def grid_search():
    """Performs grid search over PPO hyperparameters"""
    # Define the hyperparameter grid
    learning_rates = [3e-4, 1e-4, 3e-5]
    n_steps_list = [256, 512, 1024]
    clip_ranges = [0.1, 0.2, 0.3]

    # Generate all combinations
    param_combinations = list(itertools.product(learning_rates, n_steps_list, clip_ranges))
    total_runs = len(param_combinations)

    print(f"\n{'='*80}")
    print(f"Starting Grid Search")
    print(f"Total combinations: {total_runs}")
    print(f"Learning rates: {learning_rates}")
    print(f"N steps: {n_steps_list}")
    print(f"Clip ranges: {clip_ranges}")
    print(f"{'='*80}\n")

    # Initialize matrix (same logic as in main)
    print(f"Matrix dimensions: {M}x{N}")
    matrix = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL, epsilon=EPSILON, base_P=BASE_MATRIX)

    need_new_matrix = (
        matrix.base_P is None or
        matrix.base_P.shape != (M, N)
    )

    if need_new_matrix:
        print(f"Generating new {M}x{N} matrix...")
        if MATRIX_MODE == "toeplitz":
            matrix.generateMatrix(
                mode="toeplitz",
                rho=TOEPLITZ_RHO,
                signed=TOEPLITZ_SIGNED,
                antisymmetric=TOEPLITZ_ANTISYMMETRIC,
                band=TOEPLITZ_BAND
            )
        else:
            matrix.generateMatrix(mode="uniform")

        update_base_matrix(matrix.base_P)
        print("Updated base_matrix.py with new BASE_MATRIX")

        import importlib
        import base_matrix
        importlib.reload(base_matrix)
        from base_matrix import BASE_MATRIX as RELOADED_BASE
        matrix.base_P = RELOADED_BASE
        print("Reloaded base matrix configuration")
    else:
        print(f"Using existing {M}x{N} matrix from config")

    # Create directory for results
    os.makedirs('models', exist_ok=True)

    # Train for each combination
    results = []
    for run_id, (lr, n_steps, clip_range) in enumerate(param_combinations, 1):
        try:
            result = train_single_config(matrix, lr, n_steps, clip_range, run_id, total_runs)
            results.append(result)
        except Exception as e:
            print(f"\nERROR in run {run_id}: {e}")
            results.append({
                "learning_rate": lr,
                "n_steps": n_steps,
                "clip_range": clip_range,
                "error": str(e)
            })

    # Save results to JSON file
    results_file = f"models/grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "grid_search_params": {
                "learning_rates": learning_rates,
                "n_steps": n_steps_list,
                "clip_ranges": clip_ranges
            },
            "matrix_config": {
                "M": M,
                "N": N,
                "MIN_VAL": MIN_VAL,
                "MAX_VAL": MAX_VAL,
                "EPSILON": EPSILON
            },
            "training_config": {
                "TIMESTEPS": TIMESTEPS,
                "N_ENVS": N_ENVS
            },
            "results": results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Grid Search Complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

    # Print summary of all runs
    print("\nSummary of all runs:")
    print(f"{'LR':<12} {'N_STEPS':<10} {'CLIP':<8} {'MODEL_PATH':<60}")
    print("-" * 100)
    for r in results:
        if "error" not in r:
            print(f"{r['learning_rate']:<12.2e} {r['n_steps']:<10} {r['clip_range']:<8.1f} {r['model_path']:<60}")
        else:
            print(f"{r['learning_rate']:<12.2e} {r['n_steps']:<10} {r['clip_range']:<8.1f} ERROR: {r['error']}")

    return results


def main():
    print(f"Matrix dimensions: {M}x{N}")
    matrix = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL, epsilon=EPSILON, base_P=BASE_MATRIX)

    need_new_matrix = (
        matrix.base_P is None or
        matrix.base_P.shape != (M, N)
    )

    if need_new_matrix:
        print(f"Generating new {M}x{N} matrix...")
        if MATRIX_MODE == "toeplitz":
            matrix.generateMatrix(
                mode="toeplitz",
                rho=TOEPLITZ_RHO,
                signed=TOEPLITZ_SIGNED,
                antisymmetric=TOEPLITZ_ANTISYMMETRIC,
                band=TOEPLITZ_BAND
            )
        else:
            matrix.generateMatrix(mode="uniform")

        update_base_matrix(matrix.base_P)
        print("Updated base_matrix.py with new BASE_MATRIX")

        import importlib
        import base_matrix
        importlib.reload(base_matrix)
        from base_matrix import BASE_MATRIX as RELOADED_BASE
        matrix.base_P = RELOADED_BASE
        print("Reloaded base matrix configuration")
    else:
        print(f"Using existing {M}x{N} matrix from config")

    def make_env():
        base_env = RandomMatrixEnv(matrix)
        if USE_MACRO_STRATEGY:
            print("Using MacroStrategyWrapper with macro_len=10")
            base_env = MacroStrategyWrapper(base_env, macro_len=10)
        return TimeLimit(base_env, max_episode_steps=2000)

    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    model = None

    if LOAD_MODEL:
        model_path = None
        if os.path.exists('models/'):
            for file in os.listdir('models/'):
                if file.endswith('.zip') and f'matrix{M}x{N}_min{MIN_VAL}_max{MAX_VAL}_epsilon{EPSILON}' in file:
                    model_path = os.path.join('models', file)
                    break

        if model_path and os.path.exists(model_path):
            print(f"Loading existing model from: {model_path}")
            model = PPO.load(model_path, env=vec_env, verbose=1)
            print("Model loaded successfully! Continuing training...")
        else:
            print("No existing model found. Starting training from scratch...")
            model = create_ppo_model(vec_env, n_envs=N_ENVS)
    else:
        print("Starting training from scratch...")
        model = create_ppo_model(vec_env, n_envs=N_ENVS)

    if USE_INITIAL_ACTION_BIAS:
        apply_initial_action_bias(model, PREFERRED_ACTION_ID, INITIAL_BIAS)
        print(f"Applied initial bias of {INITIAL_BIAS} to action ID {PREFERRED_ACTION_ID}")
    else:
        print("Initial action bias disabled (USE_INITIAL_ACTION_BIAS = False)")

    callbacks = [EpisodeCounterCallback()]

    if USE_BIAS_ANNEALING:
        bias_callback = LogitBiasAnnealCallback(
            preferred_id=PREFERRED_ACTION_ID,
            initial_bias=INITIAL_BIAS,
            half_life=200_000
        )
        callbacks.append(bias_callback)
        print("Using LogitBiasAnnealCallback for bias annealing")

    if USE_HISTORY_TRACKING:
        history_callback = HistoryTrackerCallback(
            history_size=HISTORY_SIZE,
            no_improve_steps=NO_IMPROVE_STEPS
        )
        callbacks.append(history_callback)
        print(f"Using HistoryTrackerCallback (history_size={HISTORY_SIZE}, no_improve_steps={NO_IMPROVE_STEPS})")

    if callbacks:
        model.learn(total_timesteps=TIMESTEPS, callback=callbacks)
    else:
        model.learn(total_timesteps=TIMESTEPS)

    filename = MODEL_NAME_TEMPLATE.format(
        steps=TIMESTEPS, m=M, n=N, min=MIN_VAL, max=MAX_VAL, eps=EPSILON
    )
    model.save(filename)
    print(f"Model saved as: {filename}")


if __name__ == "__main__":
    import sys
    
    # Check command line argument to run grid search
    if len(sys.argv) > 1 and sys.argv[1] == "--grid-search":
        grid_search()
    else:
        main()


