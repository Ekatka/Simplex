import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from simplex_solver import change_to_zero_sum_phase2_only, SecondPhasePivotingEnv
from matrix import Matrix

from config import M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS, N_ENVS, MODEL_NAME_TEMPLATE, NUM_PIVOT_STRATEGIES
from base_matrix import BASE_MATRIX


class RandomMatrixEnv(SecondPhasePivotingEnv):
    def __init__(self, matrix: Matrix):
        self.matrix = matrix
        self.epsilon = matrix.epsilon
        self.K = None
        self.nit = 0        

        self._init_env()
        
        super().__init__(self.T, self.basis)

    def _init_env(self, seed=None):
        self.nit = 0
        max_attempts = 20 
        for attempt in range(max_attempts):
            try:
                perturbed_P = self.matrix.generate_perturbed_matrix()
                npMatrix = perturbed_P.base_P
            
                res = change_to_zero_sum_phase2_only(npMatrix)
                if res is not None:
                    self.T, self.basis, self.K = res
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


def update_base_matrix(matrix_data):
    """
    Updates the base_matrix.py file with new matrix data
    """
    matrix_content = []
    matrix_content.append("import numpy as np\n\n")
    matrix_content.append("# Base matrix for simplex algorithm testing\n")
    matrix_content.append("# This matrix is generated automatically during training\n")
    matrix_content.append("BASE_MATRIX = np.array([\n")
    
    for i, row in enumerate(matrix_data):
        row_str = "    [" + ", ".join(f"{val:.3f}" for val in row) + "]"
        if i == len(matrix_data) - 1:
            matrix_content.append(f"{row_str}\n")
        else:
            matrix_content.append(f"{row_str},\n")
    
    matrix_content.append("])\n")
    
    with open('base_matrix.py', 'w') as f:
        f.writelines(matrix_content)

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

        update_base_matrix(matrix.base_P)
        print("Updated base_matrix.py with new BASE_MATRIX")
        
        import importlib
        import base_matrix
        importlib.reload(base_matrix)
        from base_matrix import BASE_MATRIX
        print("Reloaded base matrix configuration")
    else:
        print(f"Using existing {M}x{N} matrix from config")
    

    vec_env = make_vec_env(lambda: RandomMatrixEnv(matrix), n_envs=N_ENVS)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=TIMESTEPS)

    filename = MODEL_NAME_TEMPLATE.format(
        steps=TIMESTEPS, m=M, n=N, min=MIN_VAL, max=MAX_VAL, eps=EPSILON
    )
    model.save(filename)

