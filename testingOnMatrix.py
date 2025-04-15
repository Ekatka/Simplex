import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trainingOnMatrix import generate_perturbed_matrix
from createMatrix import Matrix
from simplex_solver import solve_zero_sum, SecondPhasePivotingEnv
import itertools
import copy
from collections import deque
from trainingOnMatrix import RandomMatrixEnv

pivot_map = {
    0: 'bland',
    1: 'largest_coefficient',
    2: 'largest_increase',
    3: 'steepest_edge'
}

def bfs_recursion(path, env, depth=0):
    max_depth = 8
    if depth >= max_depth:
        return None

    for action in range(4):

        env_copy = copy.deepcopy(env)

        obs_next, reward, done, truncated, info = env_copy.step(action)
        new_path = path + [action]

        if done:
            return new_path

        result = bfs_recursion(new_path, env_copy, depth + 1)
        if result:
            return result

    return None

def bfs_search(env_start):

    queue = deque()
    queue.append((env_start, []))
    max_depth = 10

    while queue:
        env, path= queue.popleft()
        depth = len(path)
        if depth >= max_depth:
            continue

        # Try each action
        for action in range(4):
            env_copy = copy.deepcopy(env)
            obs_next, reward, done, truncated, info = env_copy.step(action)
            new_path = path + [action]

            if done:
                # Found a solution
                return new_path

            queue.append((env_copy, new_path,))

    return None

def find_optimal_pivot_sequence_bfs(matrix):

 # 0: bland, 1: coeff, 2: increase, 3: steepest
    pivot_map = {
        0: 'bland',
        1: 'largest_coefficient',
        2: 'largest_increase',
        3: 'steepest_edge'
    }

    done = False
    env = RandomMatrixEnv(matrix,0)

    obs, _ = env.reset()
    new_path = bfs_search(env)

    if new_path:
        final_env = RandomMatrixEnv(copy.deepcopy(matrix),0)
        obs, _ = final_env.reset()
        for a in new_path:
            obs, _, _, _, _ = final_env.step(a)
        game_value = -final_env.env.T[-1, -1]

        readable = [pivot_map[a] for a in new_path]
        print(f"\n[BFS] hortest path found in {len(new_path)} steps:")
        print(" → ".join(readable))
        print(f"[BFS] Game Value: {game_value:.6f}")
        return new_path, len(new_path), game_value

    print("[BFS] No valid solution found within search limit.")
    return None, None, None



def testing(base_matrix):


    new_P = base_matrix
    for fixed_action in range(4):
        run_fixed_strategy(new_P, fixed_action)



def run_fixed_strategy(game_matrix, fixed_action):
    env = RandomMatrixEnv(game_matrix, 0)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(fixed_action)

    method = pivot_map[fixed_action]
    print(f"[{method.title()} Pivot] Pivot steps: {env.nit}")
    print(f"[{method.title()} Pivot] Game value (v): {-env.env.T[-1, -1]:.6f}")


def extract_optimal_strategy(T, basis, m):

    num_constraints = T.shape[0] - 2
    x = np.zeros(m)
    for row in range(num_constraints):
        basic_var = int(basis[row])
        if basic_var < m:
            x[basic_var] = T[row, -1]
    total = np.sum(x)
    if total > 0:
        x = x / total
    return x




def testing_rl(base_matrix):
    """
    Test the RL policy on the composite environment.
    Uses RandomMatrixEnv from trainingOnMatrix (which automatically transitions between phases)
    and loads the trained PPO model.
    """
    print("\n--- Testing RL Policy on Composite Environment ---")
    new_P = base_matrix  # Use the same matrix.
    print("Testing Matrix P:")
    print(pd.DataFrame(new_P).to_string(index=False, header=False))
    print(new_P.tolist())

    env = RandomMatrixEnv(base_matrix, epsilon=0)  # set epsilon=0 to avoid perturbation
    model = PPO.load("ppo_simplex_random_100_000_steps_10x10")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        chosen_method = pivot_map.get(int(action), 'bland')
        print(f"[RL Policy] Chosen pivot method: {chosen_method}")
        obs, reward, done, truncated, info = env.step(action)
    final_value = -env.env.T[-1, -1] if env.phase == 2 else np.nan
    # print(f"[RL Policy] RL policy completed. Final game value: {final_value:.6f}")
    optimal_strategy = extract_optimal_strategy(env.env.T, env.env.basis, m)
    numSteps = env.nit

    print(f"[RL Policy] RL policy completed. Final game value: {final_value:.6f}")
    print("[RL Policy] Optimal mixed strategy for the row player:")
    print(optimal_strategy)

    print(f"[RL Policy] Number of steps: {numSteps}")


if __name__ == "__main__":
    matrix = Matrix()
    base_P = matrix.generateMatrix()
    m, n = matrix.returnSize()
    print("Base matrix:")
    print(pd.DataFrame(base_P).to_string(index=False, header=False))
    epsilon = matrix.returnEpsilon()
    testing_matrix = generate_perturbed_matrix(base_P, epsilon)

    testing_rl(testing_matrix)

    testing(testing_matrix)
    find_optimal_pivot_sequence_bfs(testing_matrix)


#TODO game value is correct, but returns incorrect strategy
