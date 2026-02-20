import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import PIVOT_MAP, NUM_PIVOT_STRATEGIES
from simplex_solver import (
    change_to_zero_sum_phase2_only,
    _pivot_col_heuristics, _pivot_row, _apply_pivot,
)
from matrix import Matrix


class SecondPhasePivotingEnv(gym.Env):
    def remove_artificial(self):
        for pivrow in [row for row in range(self.basis.size)
                       if self.basis[row] > self.T.shape[1] - 2]:
            non_zero_row = [col for col in range(self.T.shape[1] - 1)
                            if abs(self.T[pivrow, col]) > self.tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(self.T, self.basis, pivrow, pivcol, self.tol)
                self.nit += 1

    def __init__(self, T, basis):
        # Core state
        self.basis = basis
        self.T = T
        self.tol = 1e-9
        self.m = self.T.shape[1] - 1
        self.remove_artificial()

        # Limits & counters
        self.maxiter = 20_000
        self.nit = 0

        # Solution buffer
        if len(self.basis[:self.m]) == 0:
            self.solution = np.empty(self.T.shape[1] - 1, dtype=np.float64)
        else:
            self.solution = np.empty(max(self.T.shape[1] - 1, max(self.basis[:self.m]) + 1),
                                     dtype=np.float64)

        # Gym spaces
        self.action_space = spaces.Discrete(NUM_PIVOT_STRATEGIES)

        # === Loop & progress bookkeeping ===
        self._seen_bases = set()
        self._last_obj = float(self.T[-1, -1])
        self._degenerate_streak = 0

        # === Reward coefficients ===
        self._step_penalty = 1.0
        self._improve_coef = 0.0
        self._degenerate_penalty = 0.0
        self._loop_penalty = 0.0
        self._success_bonus = 50.0
        self._improve_tol = 1e-12

        # === Sizes and last action ===
        self._n_vars = self.T.shape[1] - 1
        self._m_rows = self.T.shape[0] - 1
        self._last_action = -1

        # === Dict observation space ===
        self.observation_space = spaces.Dict({
            "tableau": spaces.Box(low=-np.inf, high=np.inf, shape=self.T.shape, dtype=np.float64),
            "basis_onehot": spaces.Box(low=0.0, high=1.0, shape=(self._n_vars,), dtype=np.float32),
            "reduced_costs": spaces.Box(low=-np.inf, high=np.inf, shape=(self._n_vars,), dtype=np.float64),
            "objective": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "delta_objective": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "nit_norm": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.complete = False

    def _basis_key(self):
        return tuple(int(i) for i in self.basis)

    def _basis_onehot(self):
        vec = np.zeros(self._n_vars, dtype=np.float32)
        for col in self.basis:
            c = int(col)
            if 0 <= c < self._n_vars:
                vec[c] = 1.0
        return vec

    def _last_action_onehot(self):
        v = np.zeros(int(NUM_PIVOT_STRATEGIES), dtype=np.float32)
        if 0 <= self._last_action < NUM_PIVOT_STRATEGIES:
            v[self._last_action] = 1.0
        return v

    def _nit_norm(self):
        return np.array([min(1.0, float(self.nit) / float(self.maxiter))], dtype=np.float32)

    def _get_obs(self):
        tableau = self.T.copy()
        rc = tableau[-1, :-1].copy()
        obj = float(tableau[-1, -1])
        delta = float(self._last_obj - obj)

        obs = {
            "tableau": tableau,
            "basis_onehot": self._basis_onehot(),
            "reduced_costs": rc,
            "objective": np.array([obj], dtype=np.float64),
            "delta_objective": np.array([delta], dtype=np.float64),
            "nit_norm": self._nit_norm(),
            # "degenerate_streak": np.array([float(self._degenerate_streak)], dtype=np.float32),
            # "loop_flag": np.array([1.0 if self._basis_key() in self._seen_bases else 0.0], dtype=np.float32),
            # "last_action_onehot": self._last_action_onehot(),
            # "shift_K": np.array([float(getattr(self, "K", 0.0) or 0.0)], dtype=np.float64),
        }
        return obs

    def reset(self, seed=None, **kwargs):
        self.nit = 0
        self._seen_bases.clear()
        self._last_obj = float(self.T[-1, -1])
        self._degenerate_streak = 0
        self._last_action = -1
        self._seen_bases.add(self._basis_key())
        return self._get_obs(), {}

    def step(self, action):
        self._last_action = int(action)
        strategy = PIVOT_MAP[self._last_action]

        reward = -self._step_penalty
        done = False
        truncated = False

        pivcol_found, pivcol = _pivot_col_heuristics(self.T, strategy=strategy, tol=self.tol)
        if not pivcol_found:
            reward += self._success_bonus
            done = True
            info = {
                "status": "optimal",
                "nit": self.nit,
                "objective": float(self.T[-1, -1]),
                "degenerate_streak": self._degenerate_streak,
            }
            return self._get_obs(), reward, done, truncated, info

        use_bland = (strategy == 'blands_rule')
        pivrow_found, pivrow = _pivot_row(self.T, self.basis, pivcol, phase=2, tol=self.tol, bland=use_bland)
        if not pivrow_found:
            done = True
            info = {
                "status": "no_pivot_row",
                "nit": self.nit,
                "objective": float(self.T[-1, -1]),
                "degenerate_streak": self._degenerate_streak,
            }
            return self._get_obs(), reward, done, truncated, info

        old_obj = float(self.T[-1, -1])
        _apply_pivot(self.T, self.basis, pivrow, pivcol, tol=self.tol)
        self.nit += 1
        new_obj = float(self.T[-1, -1])

        delta = old_obj - new_obj
        if delta > self._improve_tol:
            reward += self._improve_coef * delta
            self._degenerate_streak = 0
        else:
            reward -= self._degenerate_penalty
            self._degenerate_streak += 1

        key = self._basis_key()
        if key in self._seen_bases:
            print("LOOP DETECTED")
            reward -= self._loop_penalty
            truncated = True
        else:
            self._seen_bases.add(key)

        if self.nit >= self.maxiter:
            done = True

        self._last_obj = new_obj

        info = {
            "status": "running",
            "nit": self.nit,
            "objective": new_obj,
            "delta_objective": delta,
            "degenerate": (delta <= self._improve_tol),
            "degenerate_streak": self._degenerate_streak,
            "pivcol": int(pivcol),
            "pivrow": int(pivrow),
            "strategy": strategy,
            "loop_detected": truncated,
        }
        return self._get_obs(), reward, done, truncated, info

    def render(self, mode='human'):
        print("Current Tableau:")
        print(self.T)
        print("Current Basis:", self.basis)
        print("Iterations:", self.nit)

    def close(self):
        pass


class FirstPhasePivotingEnv(gym.Env):

    def __init__(self, T, basis, nit0 = 0):
        self.basis = basis
        self.T = T
        self.tol = 1e-9
        self.maxiter = 5000
        self.nit0 = nit0
        self.nit = 0
        self.action_space = spaces.Discrete(NUM_PIVOT_STRATEGIES)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.T.shape, dtype=np.float64
        )
        self.complete = False

    def _get_obs(self):
        return self.T.copy()

    def reset(self, seed=None, **kwargs):
        self.nit = 0
        self.status = None
        return self._get_obs(), {}

    def step(self, action):
        strategy = PIVOT_MAP[int(action)]
        reward = -1
        done = False

        pivcol_found, pivcol = _pivot_col_heuristics(self.T, strategy=strategy, tol=self.tol)
        if not pivcol_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        pivrow_found, pivrow = _pivot_row(self.T, self.basis, pivcol, phase=1, tol=self.tol)
        if not pivrow_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        _apply_pivot(self.T, self.basis, pivrow, pivcol, tol=self.tol)
        self.nit += 1

        if self.nit >= self.maxiter:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human'):
        print("Current Tableau:")
        print(self.T)
        print("Current Basis:", self.basis)
        print("Iteration:", self.nit)

    def close(self):
        pass


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
        return obs, reward, done, truncated, info


