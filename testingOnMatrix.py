import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trainingOnMatrix import generate_perturbed_matrix
from createMatrix import Matrix
from simplex_solver import solve_zero_sum, PivotingEnv

pivot_map = {
    0: 'bland',
    1: 'largest_coefficient',
    2: 'largest_increase',
    3: 'steepest_edge'
}

def testing(base_matrix, epsilon=0.1):


    new_P = generate_perturbed_matrix(base_matrix, epsilon)
    print(f"\n[New Evaluation] Testing on matrix P':\n")
    df = pd.DataFrame(new_P)
    print(df.to_string(index=False, header=False))
    T, basis = solve_zero_sum(new_P)
    env = PivotingEnv(T, basis)

    model = PPO.load("ppo_simplex_random_10x10")
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        chosen_method = pivot_map.get(int(action), 'bland')
        print(f"[RL Policy] Chosen pivot method: {chosen_method}")
        obs, reward, done, truncated, info = env.step(action)

    print(f"[RL Policy] Pivot steps: {env.nit}")
    print(f"[RL Policy] Game value (v): {-env.T[-1, -1]:.6f}")

    for fixed_action in range(4):
        run_fixed_strategy(new_P, fixed_action)

def run_fixed_strategy(game_matrix, fixed_action):
    result = solve_zero_sum(game_matrix)
    if result is None:
        print(f"[Fixed Strategy {pivot_map[fixed_action]}] Phase 1 failed.")
        return

    T, basis = result
    env = PivotingEnv(T, basis)

    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(fixed_action)

    method = pivot_map[fixed_action]
    print(f"[{method.title()} Pivot] Pivot steps: {env.nit}")
    print(f"[{method.title()} Pivot] Game value (v): {-env.T[-1, -1]:.6f}")


if __name__ == "__main__":
    matrix = Matrix()
    base_P = matrix.generateMatrix()
    m, n = matrix.returnSize()

    testing(base_P, epsilon=0.1)

