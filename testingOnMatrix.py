import numpy as np
from stable_baselines3 import PPO
import my_simplex
from trainingOnMatrix import generate_perturbed_matrix
from createMatrix import Matrix
import pandas as pd

def testing(base_matrix, epsilon=0.1):
    pivot_map = {0: 'bland', 1: 'largest_coefficient', 2: 'largest_increase', 3: 'steepest_edge'}

    new_P = generate_perturbed_matrix(base_matrix, epsilon)
    print(f"\n[New Evaluation] Testing on matrix P':\n")
    # print(np.array2string(new_P, separator=", ", formatter={'int': lambda x: f"{x:10d}"}))

    df = pd.DataFrame(new_P)
    print(df.to_string(index=False, header=False))

    A = np.hstack([-new_P.T, np.ones((n, 1))])
    b = np.zeros(n)
    c = np.hstack([np.zeros(m), -1])
    one_row = np.hstack([np.ones(m), [0]])
    A = np.vstack([A, one_row])
    b = np.append(b, [1])

    eval_env = my_simplex.SimplexGymEnv(A, b, c, maxiter=5000)

    model = PPO.load("ppo_simplex_random_10x10")


    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        chosen_method = pivot_map.get(int(action), 'bland')
        print(f"[New Evaluation] Chosen method: {chosen_method}")
        obs, reward, done, truncated, info = eval_env.step(action)

    pivot_steps_rl = eval_env.nit
    print(f"[RL Policy] Pivot steps: {pivot_steps_rl}")

    pivot_steps_bland = run_fixed_strategy(my_simplex.SimplexGymEnv(A, b, c, maxiter=5000), fixed_action=0)
    pivot_steps_coefficient = run_fixed_strategy(my_simplex.SimplexGymEnv(A, b, c, maxiter=5000), fixed_action=1)
    pivot_steps_increase = run_fixed_strategy(my_simplex.SimplexGymEnv(A, b, c, maxiter=5000), fixed_action=2)
    pivot_steps_steepest = run_fixed_strategy(my_simplex.SimplexGymEnv(A, b, c, maxiter=5000), fixed_action=3)

    print(f"[Bland Pivot] Pivot steps: {pivot_steps_bland}")
    print(f"[Coefficient Pivot] Pivot steps: {pivot_steps_coefficient}")
    print(f"[Increase Pivot] Pivot steps: {pivot_steps_increase}")
    print(f"[Steepest Edge Pivot] Pivot steps: {pivot_steps_steepest}")


def run_fixed_strategy(env, fixed_action):
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(fixed_action)
    return env.nit


if __name__ == "__main__":
    matrix = Matrix()
    base_P = matrix.generateMatrix()
    m, n = matrix.returnSize()
    testing(base_P, 0.1)
