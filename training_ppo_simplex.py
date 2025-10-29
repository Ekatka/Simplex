import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
from gymnasium.wrappers import TimeLimit
from simplex_solver import change_to_zero_sum_phase2_only, SecondPhasePivotingEnv
from matrix import Matrix

from config import M, N, MIN_VAL, MAX_VAL, EPSILON, TIMESTEPS, N_ENVS, MODEL_NAME_TEMPLATE, NUM_PIVOT_STRATEGIES, LOAD_MODEL, PREFERRED_ACTION_ID, INITIAL_BIAS, USE_MACRO_STRATEGY, USE_BIAS_ANNEALING, USE_INITIAL_ACTION_BIAS, USE_HISTORY_TRACKING, HISTORY_SIZE, NO_IMPROVE_STEPS
from base_matrix import BASE_MATRIX

from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import math

from stable_baselines3.common.callbacks import BaseCallback

class EarlyStopWrapper(gym.Wrapper):
    def __init__(self, env, max_degenerate_streak=100, window=200, improve_tol=1e-12):
        super().__init__(env)
        self.max_degenerate_streak = int(max_degenerate_streak)
        self.window = int(window)
        self.improve_tol = float(improve_tol)
        self._last_obj = None
        self._no_improve_steps = 0

    def reset(self, **kwargs):
        self._last_obj = None
        self._no_improve_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)

        # track objective improvement
        obj = float(info.get("objective", self.env.T[-1, -1]))
        if self._last_obj is None:
            self._last_obj = obj
        else:
            delta = self._last_obj - obj
            if delta > self.improve_tol:
                self._no_improve_steps = 0
            else:
                self._no_improve_steps += 1
            self._last_obj = obj

        # early stop on long degeneracy
        if info.get("degenerate_streak", 0) >= self.max_degenerate_streak:
            truncated = True

        # early stop on stagnation window
        if self._no_improve_steps >= self.window:
            truncated = True

        return obs, rew, done, truncated, info

class EpisodeCounterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.completed_this_iter = 0

    def _on_rollout_start(self) -> None:
        self.completed_this_iter = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:  # Monitor puts this when an episode ends
                self.completed_this_iter += 1
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("debug/episodes_finished_in_rollout", self.completed_this_iter)


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


class HistoryTrackerCallback(BaseCallback):
    """
    Tracks history of the last N steps and prints it when the objective function doesn't improve.
    Tracks only the first env from vectorized environment.
    """
    def __init__(self, history_size: int = 20, no_improve_steps: int = 100, improve_tol: float = 1e-12):
        super().__init__()
        self.history_size = int(history_size)
        self.no_improve_steps = int(no_improve_steps)
        self.improve_tol = float(improve_tol)

        # History of last steps
        self.history = []
        self._last_obj = None
        self._no_improve_counter = 0
        self._printed_this_episode = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])
        rewards = self.locals.get("rewards", [])

        # Track only the first env (index 0)
        if infos and len(infos) > 0:
            info = infos[0]

            # If episode completed - reset for next episode
            if "episode" in info:
                self._last_obj = None
                self._no_improve_counter = 0
                self._printed_this_episode = False
                self.history = []
                return True

            # Get information about current step
            obj = info.get("objective")
            action = actions[0] if actions is not None and len(actions) > 0 else None
            reward = rewards[0] if rewards is not None and len(rewards) > 0 else None
            strategy = info.get("strategy", "unknown")
            degenerate = info.get("degenerate", False)
            nit = info.get("nit", 0)

            # Add step to history
            step_info = {
                "nit": nit,
                "objective": obj,
                "action": action,
                "strategy": strategy,
                "reward": reward,
                "degenerate": degenerate
            }
            self.history.append(step_info)

            # Keep only last history_size steps
            if len(self.history) > self.history_size:
                self.history.pop(0)

            # Check objective function improvement
            if self._last_obj is not None and obj is not None:
                delta = self._last_obj - obj
                if delta > self.improve_tol:
                    # Improvement detected - reset counter
                    self._no_improve_counter = 0
                else:
                    # No improvement - increment counter
                    self._no_improve_counter += 1

            if obj is not None:
                self._last_obj = obj

            # Print history if no improvement and not yet printed in this episode
            if self._no_improve_counter >= self.no_improve_steps and not self._printed_this_episode:
                self._print_history()
                self._printed_this_episode = True

        return True

    def _print_history(self):
        """
        Prints the history of last steps
        """
        if not self.history:
            return

        print("\n" + "="*80)
        print(f"Objective hasn't improved for {self._no_improve_counter} steps")
        print(f"Last {len(self.history)} steps:")
        print("="*80)
        print(f"{'Step':<8} {'Objective':<15} {'Action':<10} {'Strategy':<20} {'Reward':<10} {'Degenerate':<12}")
        print("-"*80)

        from config import PIVOT_MAP

        for i, step in enumerate(self.history):
            nit = step.get("nit", "N/A")
            obj = step.get("objective", "N/A")
            action = step.get("action", "N/A")
            strategy = step.get("strategy", "N/A")
            reward = step.get("reward", "N/A")
            degenerate = step.get("degenerate", False)

            # Format values
            if isinstance(obj, (int, float)):
                obj_str = f"{obj:.8e}"
            else:
                obj_str = str(obj)

            if isinstance(action, (int, np.integer)):
                action_name = PIVOT_MAP.get(int(action), f"act_{action}")
            else:
                action_name = str(action)

            if isinstance(reward, (int, float)):
                reward_str = f"{reward:.4f}"
            else:
                reward_str = str(reward)

            degenerate_str = "Yes" if degenerate else "No"

            print(f"{nit:<8} {obj_str:<15} {action_name:<10} {strategy:<20} {reward_str:<10} {degenerate_str:<12}")

        print("="*80 + "\n")


def apply_initial_action_bias(model, preferred_id: int, initial_bias: float = 3.0):
    with th.no_grad():
        bias = model.policy.action_net.bias
        bias[:] = 0.0
        bias[int(preferred_id)] = float(initial_bias)


# def create_ppo_model(vec_env, verbose=1)
#     return PPO(
#         "MlpPolicy",
#         vec_env,
#         verbose=verbose,
#         gamma=0.999,          # longer credit assignment
#         n_steps=2048,         # increase if memory allows
#         batch_size=4096//N_ENVS,
#         ent_coef=0.0,         # reduce random dithering; exploration via bias
#         learning_rate=3e-4,
#         clip_range=0.2,
#     )
def create_ppo_model(vec_env, verbose=1):
    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        gamma=0.995,          # slightly shorter horizon
        n_steps=1024,         # more frequent updates -> more episode finishes in a rollout
        batch_size=max(64, 2048//N_ENVS),
        ent_coef=0.01,        # restore exploration
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
            base_env = MacroStrategyWrapper(base_env, macro_len=10)
        # HARD CAP: e.g. 2000 pivots -> truncated=True if hit
        return TimeLimit(base_env, max_episode_steps=2000)

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

    if USE_INITIAL_ACTION_BIAS:
        apply_initial_action_bias(model, PREFERRED_ACTION_ID, INITIAL_BIAS)
        print(f"Applied initial bias of {INITIAL_BIAS} to action ID {PREFERRED_ACTION_ID}")
    else:
        print("Initial action bias disabled (USE_INITIAL_ACTION_BIAS = False)")

    # Create callbacks
    callbacks = [EpisodeCounterCallback()]
    
    if USE_BIAS_ANNEALING:
        bias_callback = LogitBiasAnnealCallback(
            preferred_id=PREFERRED_ACTION_ID,
            initial_bias=INITIAL_BIAS,
            half_life=200_000
        )
        callbacks.append(bias_callback)
        print("Using LogitBiasAnnealCallback for bias annealing")
    
    if USE_HISTORY_TRACKING:
        history_callback = HistoryTrackerCallback(
            history_size=HISTORY_SIZE,
            no_improve_steps=NO_IMPROVE_STEPS
        )
        callbacks.append(history_callback)
        print(f"Using HistoryTrackerCallback (history_size={HISTORY_SIZE}, no_improve_steps={NO_IMPROVE_STEPS})")
    
       
    if callbacks:
        model.learn(total_timesteps=TIMESTEPS, callback=callbacks)
    else:
        model.learn(total_timesteps=TIMESTEPS)

    filename = MODEL_NAME_TEMPLATE.format(
        steps=TIMESTEPS, m=M, n=N, min=MIN_VAL, max=MAX_VAL, eps=EPSILON
    )
    model.save(filename)
    print(f"Model saved as: {filename}")

