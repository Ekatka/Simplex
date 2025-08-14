import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from simplex_solver import change_to_zero_sum_phase2_only, SecondPhasePivotingEnv
from matrix import Matrix

from config import M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS, N_ENVS, BASE_MATRIX, MODEL_NAME_TEMPLATE, NUM_PIVOT_STRATEGIES


class RandomMatrixEnv(SecondPhasePivotingEnv):
    def __init__(self, matrix: Matrix):
        self.matrix = matrix  # Store full Matrix object
        self.epsilon = matrix.epsilon
        self.K = None  # Store the shift constant K
        self.nit = 0

        self._init_env()
        
        # Call parent constructor with the created tableau and basis
        super().__init__(self.T, self.basis)

    def _init_env(self, seed=None):
        self.nit = 0
        max_attempts = 20 
        for attempt in range(max_attempts):
            try:
                perturbed_P = self.matrix.generate_perturbed_matrix()
                npMatrix = perturbed_P.base_P
                # takes matrix and makes a zero sum game linear program out of it
                # Now using direct Phase 2 approach - no Phase 1 needed
                res = change_to_zero_sum_phase2_only(npMatrix)
                # if it is not possible to do - numerical instability or some different problem try to do a new perturbed matrix
                if res is not None:
                    # if the solution was found start directly with Phase 2
                    self.T, self.basis, self.K = res  # Store K for game value calculation
                    return
            except Exception as e:
                print(f"[RandomMatrixEnv] Attempt {attempt + 1} failed: {e}")
                continue

        print(f"Too many unstable matrices for size {self.matrix.m}x{self.matrix.n}")
        raise RuntimeError("Failed to initialize a stable Phase 2 tableau.")

    def reset(self, seed=None, **kwargs):
        self._init_env(seed)
        self.nit = 0
        return super().reset(seed=seed)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.nit += 1
        return obs, reward, done, truncated, info


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

