import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from training_ppo_simplex import RandomMatrixEnv
from matrix import Matrix

epsilon = 0
trainingSize = 10_000


def trainingTimeForMatrix(matrix_size):

    matrix_instance = Matrix()
    matrix_instance.resize(matrix_size, matrix_size)
    base_P = matrix_instance.generateMatrix()

    vec_env = make_vec_env(lambda: RandomMatrixEnv(base_P), n_envs=4)

    model = PPO("MlpPolicy", vec_env, verbose=0)

    start_time = time.perf_counter()
    model.learn(total_timesteps=trainingSize)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    return elapsed


def main():
    matrix_sizes = list(range(10, 110, 10))
    times = []

    print("Measuring training time for one iteration (", trainingSize, " timesteps) versus matrix size:")
    for size in matrix_sizes:
        t = trainingTimeForMatrix(size)
        times.append(t)
        print(f"Matrix size {size}x{size}: {t:.4f} seconds")

    # Plot the results.
    plt.figure(figsize=(8, 5))
    plt.plot(np.square(matrix_sizes), times, marker="o")
    plt.xlabel("Matrix Size (m x m)")
    plt.ylabel("Training Time per Iteration (seconds)")
    plt.title(f"Training Time vs Matrix Size (One Iteration = {trainingSize} timesteps)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
