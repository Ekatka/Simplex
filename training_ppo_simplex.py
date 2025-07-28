import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from simplex_solver import change_to_zero_sum, FirstPhasePivotingEnv, SecondPhasePivotingEnv, first_to_second
from matrix import Matrix

from config import M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS, N_ENVS, BASE_MATRIX, MODEL_NAME_TEMPLATE



def pad_observation(obs, target_shape):
    padded = np.zeros(target_shape, dtype=obs.dtype)
    padded[:obs.shape[0], :obs.shape[1]] = obs
    return padded


class RandomMatrixEnv(gym.Env):
    def __init__(self, matrix: Matrix):
        super().__init__()
        self.matrix = matrix  # Store full Matrix object
        self.epsilon = matrix.epsilon
        self.env = None
        self.phase = None
        self.av = None
        self.nit = 0

        self._init_env()
        self.fixed_obs_shape = self.env.observation_space.shape
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _init_env(self, seed=None):
        self.nit = 0
        max_attempts = 20 
        for attempt in range(max_attempts):
            try:
                perturbed_P = self.matrix.generate_perturbed_matrix()
                npMatrix = perturbed_P.base_P
                # takes matrix and makes a zero sum game linear program out of it
                res = change_to_zero_sum(npMatrix)
                # if it is not possible to do - numerical instability or some different problem try to do a new perturbed matrix
                if res is not None:
                    # if the solution was found start first phase
                    T, basis, self.av = res
                    self.phase = 1
                    self.env = FirstPhasePivotingEnv(T, basis)
                    if seed is not None:
                        self.env.reset(seed=seed)
                    return
            except Exception as e:
                print(f"[RandomMatrixEnv] Attempt {attempt + 1} failed: {e}")
                continue

        print(f"Too many unstable matrices for size {self.matrix.m}x{self.matrix.n}")
        raise RuntimeError("Failed to initialize a stable Phase 1 tableau.")

    def reset(self, seed=None, **kwargs):
        self._init_env(seed)
        obs, info = self.env.reset(seed=seed)
        return pad_observation(obs, self.fixed_obs_shape), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.nit += 1

        if self.phase == 1 and done:
            # if the first phase is done, start the second
            result = first_to_second(self.env.T, self.env.basis, self.av)
            if result is None:
                # Phase 1 failed - restart with a new matrix
                print(f"[RandomMatrixEnv] Phase 1 failed for matrix size {self.matrix.m}x{self.matrix.n}, restarting...")
                self._init_env()
                obs, _ = self.env.reset()
                done = False
                reward = -0.1
            else:
                T2, basis2 = result
                self.env = SecondPhasePivotingEnv(T2, basis2)
                self.phase = 2
                obs, _ = self.env.reset()
                done = False

        return pad_observation(obs, self.fixed_obs_shape), reward, done, truncated, info


def update_config_with_matrix(matrix_data):

    config_content = []
    with open('config.py', 'r') as f:
        for line in f:
            if line.strip().startswith('BASE_MATRIX ='):
                config_content.append('BASE_MATRIX = np.array([\n')
                for i, row in enumerate(matrix_data):
                    row_str = "    [" + ", ".join(f"{val:.3f}" for val in row) + "]"
                    if i == len(matrix_data) - 1:
                        config_content.append(f"{row_str}\n")
                    else:
                        config_content.append(f"{row_str},\n")
                config_content.append('])\n')
            else:
                config_content.append(line)
    
    with open('config.py', 'w') as f:
        f.writelines(config_content)

if __name__ == "__main__":
    print(M,N)
    matrix = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL, epsilon=EPSILON, base_P=BASE_MATRIX)


    need_new_matrix = (
        matrix.base_P is None or 
        matrix.base_P.shape != (M, N)
    )
    
    if need_new_matrix:
        print(f"Generating new {M}x{N} matrix...")
        matrix.generateMatrix()

        update_config_with_matrix(matrix.base_P)
        print("Updated config.py with new BASE_MATRIX")
        
        import importlib
        import config
        importlib.reload(config)
        from config import BASE_MATRIX
        print("Reloaded configuration")
    else:
        print(f"Using existing {M}x{N} matrix from config")
    

    vec_env = make_vec_env(lambda: RandomMatrixEnv(matrix), n_envs=N_ENVS)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=TIMESTEPS)

    filename = MODEL_NAME_TEMPLATE.format(
        steps=TIMESTEPS, m=M, n=N, min=MIN_VAL, max=MAX_VAL, eps=EPSILON
    )
    model.save(filename)

