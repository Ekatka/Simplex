import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

from simplex_solver import change_to_zero_sum_phase2_only, SecondPhasePivotingEnv
from matrix import Matrix

from config import M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS, N_ENVS, MODEL_NAME_TEMPLATE, NUM_PIVOT_STRATEGIES, LOAD_MODEL, PREFERRED_ACTION_ID, INITIAL_BIAS, USE_MACRO_STRATEGY, USE_BIAS_ANNEALING
from base_matrix import BASE_MATRIX

from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import math


class MacroStrategyWrapper(gym.Wrapper):
    """
    Agent chooses a strategy id; env repeats it for next `macro_len` pivots.
    """
    def __init__(self, env, macro_len: int = 10):
        super().__init__(env)
        self.macro_len = macro_len
        self._remaining = 0
        self._current_strategy = None

    def reset(self, **kwargs):
        self._remaining = 0
        self._current_strategy = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._remaining == 0:
            self._current_strategy = int(action)
            self._remaining = self.macro_len
        obs, reward, terminated, truncated, info = self.env.step(self._current_strategy)
        self._remaining -= 1
        info = dict(info)
        info["macro_strategy"] = self._current_strategy
        info["macro_remaining"] = self._remaining
        return obs, reward, terminated, truncated, info


class LogitBiasAnnealCallback(BaseCallback):
    def __init__(self, preferred_id: int, initial_bias: float = 3.0, half_life: int = 500_000, verbose=0):
        super().__init__(verbose)
        self.preferred_id = int(preferred_id)
        self.initial_bias = float(initial_bias)
        self.half_life = int(half_life)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        factor = math.pow(0.5, t / max(1, self.half_life))  # exponential decay
        with th.no_grad():
            bias = self.model.policy.action_net.bias  # shape [num_actions]
            bias[:] = 0.0
            bias[self.preferred_id] = self.initial_bias * factor
        return True


def apply_initial_action_bias(model, preferred_id: int, initial_bias: float = 3.0):
    with th.no_grad():
        bias = model.policy.action_net.bias
        bias[:] = 0.0
        bias[int(preferred_id)] = float(initial_bias)


def create_ppo_model(vec_env, verbose=1):
    """Создает PPO модель с одинаковыми параметрами во всех случаях"""
    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        gamma=0.999,          # longer credit assignment
        n_steps=2048,         # increase if memory allows
        batch_size=4096//N_ENVS,
        ent_coef=0.0,         # reduce random dithering; exploration via bias
        learning_rate=3e-4,
        clip_range=0.2,
    )


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
    print(f"Matrix dimensions: {M}x{N}")
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
    
    # Create environment with the possibility of using MacroStrategyWrapper
    def make_env():
        base_env = RandomMatrixEnv(matrix)
        if USE_MACRO_STRATEGY:
            print("Using MacroStrategyWrapper with macro_len=10")
            return MacroStrategyWrapper(base_env, macro_len=10)
        return base_env
    
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    
    # Initialize model in all cases
    model = None
    
    if LOAD_MODEL:
        # Search for existing model
        model_path = None
        for file in os.listdir('models/'):
            if file.endswith('.zip') and f'matrix{M}x{N}_min{MIN_VAL}_max{MAX_VAL}_epsilon{EPSILON}' in file:
                model_path = os.path.join('models', file)
                break
        
        if model_path and os.path.exists(model_path):
            print(f"Loading existing model from: {model_path}")
            model = PPO.load(model_path, env=vec_env, verbose=1)
            print("Model loaded successfully! Continuing training...")
        else:
            print("No existing model found. Starting training from scratch...")
            model = create_ppo_model(vec_env)
    else:
        print("Starting training from scratch...")
        model = create_ppo_model(vec_env)
    
    # Apply bias in all cases
    apply_initial_action_bias(model, PREFERRED_ACTION_ID, INITIAL_BIAS)
    
    # Create callback for annealing bias if used
    callbacks = []
    if USE_BIAS_ANNEALING:
        bias_callback = LogitBiasAnnealCallback(
            preferred_id=PREFERRED_ACTION_ID,
            initial_bias=INITIAL_BIAS,
            half_life=500_000
        )
        callbacks.append(bias_callback)
        print("Using LogitBiasAnnealCallback for bias annealing")
    
       
    if callbacks:
        model.learn(total_timesteps=TIMESTEPS, callback=callbacks)
    else:
        model.learn(total_timesteps=TIMESTEPS)

    filename = MODEL_NAME_TEMPLATE.format(
        steps=TIMESTEPS, m=M, n=N, min=MIN_VAL, max=MAX_VAL, eps=EPSILON
    )
    model.save(filename)
    print(f"Model saved as: {filename}")

