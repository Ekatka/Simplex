import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from matrix import Matrix
from simplex_solver import change_to_zero_sum_phase2_only, SecondPhasePivotingEnv
from training_ppo_simplex import RandomMatrixEnv
from collections import deque
import copy

# Import constants
from config import (
    M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS,
    BASE_MATRIX, MODEL_NAME_TEMPLATE, BFS_DEPTH, PIVOT_MAP, NUM_PIVOT_STRATEGIES
)
def bfs_search(env_start):
    queue = deque()
    queue.append((env_start, []))
    while queue:
        env, path = queue.popleft()
        if len(path) >= BFS_DEPTH:
            continue
        for action in range(NUM_PIVOT_STRATEGIES):  # Use constant from config
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
        
        # Extract strategies
        first_player_strategy = extract_optimal_strategy(final_env.T, final_env.basis, M)
        second_player_strategy = extract_second_player_strategy(final_env.T, final_env.basis, M, N)
        
        # Compute game value using the correct formula from game theory
        game_value = compute_game_value_from_strategies(matrix, first_player_strategy, second_player_strategy)
        
        readable = [PIVOT_MAP[a] for a in path]
        print(f"\n[BFS] Shortest path: {' → '.join(readable)} ({len(path)} steps)")
        print(f"[BFS] Game Value: {game_value:.6f}")
        print(f"[BFS] First Player Strategy: {first_player_strategy}")
        print(f"[BFS] Second Player Strategy: {second_player_strategy}")
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
    
    # Extract strategies
    first_player_strategy = extract_optimal_strategy(env.T, env.basis, M)
    second_player_strategy = extract_second_player_strategy(env.T, env.basis, M, N)
    
    # Compute game value using the correct formula from game theory
    game_value = compute_game_value_from_strategies(matrix, first_player_strategy, second_player_strategy)
    
    print(f"[{method.title()} Pivot] Steps: {env.nit}, Game Value: {game_value:.6f}")

def test_fixed_strategies(matrix: Matrix):
    for action in range(NUM_PIVOT_STRATEGIES):  # Use constant from config
        run_fixed_strategy(matrix, action)

def extract_optimal_strategy(T, basis, m):

    num_constraints = T.shape[0] - 1 
    num_vars = T.shape[1] - 1  
    

    x = np.zeros(num_vars)
    

    for row in range(num_constraints):
        var = basis[row]  
        if 0 <= var < num_vars:  
            x[var] = T[row, -1]  
    strategy = x[:m]
    
    total = strategy.sum()
    if total > 1e-8: 
        strategy = strategy / total
    else:
        strategy = np.ones(m) / m
    
    return strategy

def extract_second_player_strategy(T, basis, m, n):

    num_constraints = T.shape[0] - 1 
    num_vars = T.shape[1] - 1 
    objective_row = T[-1, :-1] 
    dual_vars = objective_row[-n:]
    dual_vars = np.abs(dual_vars)
    total = dual_vars.sum()
    if total > 1e-8:
        second_player_strategy = dual_vars / total
    else:
        second_player_strategy = np.ones(n) / n
    
    return second_player_strategy

def compute_game_value_from_strategies(matrix: Matrix, first_player_strategy, second_player_strategy):
    """
    Compute the game value using the optimal strategies and the original payoff matrix.
    For zero-sum games: Game Value = x^T * P * y
    
    Args:
        matrix: The original payoff matrix
        first_player_strategy: Optimal strategy for first player (row player)
        second_player_strategy: Optimal strategy for second player (column player)
    
    Returns:
        game_value: The computed game value
    """
    # Convert strategies to numpy arrays if they aren't already
    x = np.array(first_player_strategy)
    y = np.array(second_player_strategy)
    
    # Get the payoff matrix
    P = matrix.base_P
    
    # Compute game value: x^T * P * y
    game_value = x.T @ P @ y
    
    return game_value


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
        # print(f"[RL] Action: {PIVOT_MAP[int(action)]}")
        obs, _, done, _, _ = env.step(action)

    # Extract strategies
    first_player_strategy = extract_optimal_strategy(env.T, env.basis, M)
    second_player_strategy = extract_second_player_strategy(env.T, env.basis, M, N)
    
    # Compute game value using the correct formula from game theory
    game_value = compute_game_value_from_strategies(matrix, first_player_strategy, second_player_strategy)
    
    print(f"[RL] Game Value: {game_value:.6f}")
    print("[RL] First Player Strategy:", first_player_strategy)
    print("[RL] Second Player Strategy:", second_player_strategy)
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
    # find_optimal_pivot_sequence_bfs(test_matrix)
