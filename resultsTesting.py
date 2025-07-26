import time
import csv
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from matrix import Matrix
from training_ppo_simplex import RandomMatrixEnv

# int to actions
pivot_map = {
    0: "bland",
    1: "largest_coefficient",
    2: "largest_increase",
    3: "steepest_edge",
}

# Configuration
EPSILON = 0.1
N_ENVS = 4
TEST_MATRICES = 5        #number of matrices average is tested on
BASE_TRAIN_STEPS = 2000  # base timesteps, will be multiplied by m^2
SIZES = list(range(2, 100, 2))  # matrix sizes: 2,4,6,8


def train_agent_for_size(m):
    """
    1) Generate one random m×m matrix.
    2) Build a vectorized RandomMatrixEnv over that matrix.
    3) Train PPO for BASE_TRAIN_STEPS * m^2 timesteps.
    Returns (model, train_time_seconds, base_matrix).
    """
    matrix = Matrix(m, m, -1, 1, EPSILON)
    matrix.generateMatrix()

    env_fn = lambda: RandomMatrixEnv(matrix)
    vec_env = make_vec_env(env_fn, n_envs=N_ENVS)

    model = PPO("MlpPolicy", vec_env, verbose=0)
    total_steps = BASE_TRAIN_STEPS * (m**2)

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_steps)
    t1 = time.perf_counter()

    return model, (t1 - t0), matrix


def run_fixed_strategy(matrix, action):
    """
    Run RandomMatrixEnv on passed matrix using fixed pivot `action
    Return total number of pivot steps (Phase 1 + Phase 2).
    """
    env = RandomMatrixEnv(matrix)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(action)
    return env.nit


def run_rl_strategy(model, matrix):
    """
    Run the composite RandomMatrixEnv under the trained
    Return total number of pivot steps (Phase 1 + Phase 2).
    """
    env = RandomMatrixEnv(matrix)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
    return env.nit


def benchmark_size(m):
    """
    For matrix size m:
      - Train an RL model (record time)
      - Generate TEST_MATRICES fresh matrices of size m×m
      - For each test matrix, record:
          * RL pivot steps
          * each fixed heuristic pivot steps
      - Return a dict of averages and train_time.
    """
    model, train_time, _ = train_agent_for_size(m)

    rl_steps = []
    fixed = {name: [] for name in pivot_map.values()}

    for _ in range(TEST_MATRICES):
        matrix = Matrix(m, m, -1, 1, EPSILON)
        matrix.generateMatrix()

        rl_steps.append(run_rl_strategy(model, matrix))
        for action, name in pivot_map.items():
            fixed[name].append(run_fixed_strategy(matrix, action))

    return {
        "m": m,
        "elements": m * m,
        "train_time": train_time,
        "avg_rl_steps": np.mean(rl_steps),
        "avg_fixed": {name: np.mean(fixed[name]) for name in pivot_map.values()}
    }


def main():
    # Prepare results container and CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_results_{timestamp}.csv"
    fieldnames = ["size", "elements", "train_time", "avg_rl_steps"] + [
        f"avg_fixed_{name}" for name in pivot_map.values()
    ]

    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print("Size | Train(s) |  RL_steps  |  Fixed(best)  [others...]")
        for m in SIZES:
            r = benchmark_size(m)
            fixed_vals = r["avg_fixed"]
            best_name = min(fixed_vals, key=lambda k: fixed_vals[k])
            best_steps = fixed_vals[best_name]
            others = ", ".join(f"{n[:3]}={fixed_vals[n]:.1f}" for n in pivot_map.values())

            print(f"{m}×{m} | {r['train_time']:.2f}s |  {r['avg_rl_steps']:.1f}   |  "
                  f"{best_name[:3]}={best_steps:.1f}  [{others}]")

            # Write one row to CSV
            row = {
                "size": r["m"],
                "elements": r["elements"],
                "train_time": r["train_time"],
                "avg_rl_steps": r["avg_rl_steps"],
            }
            for name, val in fixed_vals.items():
                row[f"avg_fixed_{name}"] = val
            writer.writerow(row)

    print(f"\nAll results written to {csv_filename}\n")

    # Reload CSV data for plotting
    sizes = []
    elements = []
    train_times = []
    avg_rl = []
    avg_fixed = {name: [] for name in pivot_map.values()}

    with open(csv_filename, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sizes.append(int(row["size"]))
            elements.append(int(row["elements"]))
            train_times.append(float(row["train_time"]))
            avg_rl.append(float(row["avg_rl_steps"]))
            for name in pivot_map.values():
                avg_fixed[name].append(float(row[f"avg_fixed_{name}"]))

    # Plot 1: Training time vs matrix elements
    plt.figure(figsize=(6, 4))
    plt.plot(elements, train_times, marker="o")
    plt.xlabel("Matrix elements (m²)")
    plt.ylabel("Training time (s)")
    plt.title(f"Training time for {BASE_TRAIN_STEPS}·m² timesteps")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"train_time_vs_size_{timestamp}.png")
    plt.close()

    # Plot 2: Average pivot steps vs matrix elements
    plt.figure(figsize=(6, 4))
    plt.plot(elements, avg_rl, label="RL", marker="o", linestyle="-")
    for name, steps in avg_fixed.items():
        plt.plot(elements, steps, label=name, marker=".", linestyle="--")
    plt.xlabel("Matrix elements (m²)")
    plt.ylabel("Average pivot steps")
    plt.title(f"Average pivot steps on {TEST_MATRICES} test matrices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"avg_pivot_steps_vs_size_{timestamp}.png")
    plt.close()

    # print("Plots saved as:")
    # print(f"  train_time_vs_size_{timestamp}.png")
    # print(f"  avg_pivot_steps_vs_size_{timestamp}.png")


if __name__ == "__main__":
    main()