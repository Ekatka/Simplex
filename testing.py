import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from matrix import Matrix
from simplex_solver import change_to_zero_sum, SecondPhasePivotingEnv
from training_ppo_simplex import RandomMatrixEnv
from collections import deque
import copy

# Import constants
from config import (
    M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS,
    BASE_MATRIX, MODEL_NAME_TEMPLATE, BFS_DEPTH, PIVOT_MAP
)
def bfs_search(env_start):
    queue = deque()
    queue.append((env_start, []))
    while queue:
        env, path = queue.popleft()
        if len(path) >= BFS_DEPTH:
            continue
        for action in range(4):
            env_copy = copy.deepcopy(env)
            _, _, done, _, _ = env_copy.step(action)
            new_path = path + [action]
            if done:
                return new_path
            queue.append((env_copy, new_path))
    return None

def find_optimal_pivot_sequence_bfs(matrix: Matrix):
    env = RandomMatrixEnv(matrix)
    _, _ = env.reset()
    path = bfs_search(env)

    if path:
        final_env = RandomMatrixEnv(matrix.copy())
        _, _ = final_env.reset()
        for a in path:
            _, _, _, _, _ = final_env.step(a)
        game_value = -final_env.env.T[-1, -1]
        readable = [PIVOT_MAP[a] for a in path]
        print(f"\n[BFS] Shortest path: {' → '.join(readable)} ({len(path)} steps)")
        print(f"[BFS] Game Value: {game_value:.6f}")
        return path, len(path), game_value

    print("[BFS] No valid solution found within search limit.")
    return None, None, None

def run_fixed_strategy(matrix: Matrix, action: int):
    env = RandomMatrixEnv(matrix)
    _, _ = env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(action)
    method = PIVOT_MAP[action]
    print(f"[{method.title()} Pivot] Steps: {env.nit}")
    print(f"[{method.title()} Pivot] Game Value: {-env.env.T[-1, -1]:.6f}")

def test_fixed_strategies(matrix: Matrix):
    for action in range(4):
        run_fixed_strategy(matrix, action)

def extract_optimal_strategy(T, basis, m):
    # num_constraints = T.shape[0] - 1  # exclude objective row
    # num_vars = T.shape[1] - 1  # exclude RHS column
    #
    # x = np.zeros(num_vars)
    #
    # for row in range(num_constraints):
    #     var = basis[row]
    #     if 0 <= var < num_vars:
    #         x[var] = T[row, -1]
    #
    # strategy = x[:m]
    # total = strategy.sum()
    # return strategy / total if total > 1e-8 else strategy
    return "Not tested"

def test_rl(matrix: Matrix):
    print("\n--- PPO Policy Evaluation ---")
    print("Matrix P:")
    print(pd.DataFrame(matrix.base_P).to_string(index=False, header=False))
    print(matrix.base_P.tolist())

    model_path = MODEL_NAME_TEMPLATE.format(
        steps=TIMESTEPS,
        m=M,
        n=N,
        min=MIN_VAL,
        max=MAX_VAL,
        eps=EPSILON
    )

    env = RandomMatrixEnv(matrix)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        print(f"[RL] Action: {PIVOT_MAP[int(action)]}")
        obs, _, done, _, _ = env.step(action)

    game_value = -env.env.T[-1, -1] if env.phase == 2 else np.nan
    # print(env.env.T)
    # print(env.env.basis)
    # print(M)
    strategy = extract_optimal_strategy(env.env.T, env.env.basis, M)
    print(f"[RL] Game Value: {game_value:.6f}")
    print("[RL] Mixed Strategy:", strategy)
    print(f"[RL] Steps Taken: {env.nit}")

if __name__ == "__main__":
    print(M,N)
    matrix = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL, epsilon=EPSILON, base_P=BASE_MATRIX)
    print("Base matrix:")
    print(pd.DataFrame(matrix.base_P).to_string(index=False, header=False))

    test_matrix = matrix.generate_perturbed_matrix()
    print("\nTesting Matrix:")
    print(pd.DataFrame(test_matrix.base_P).to_string(index=False, header=False))

    test_rl(test_matrix)
    test_fixed_strategies(test_matrix)
    find_optimal_pivot_sequence_bfs(test_matrix)
