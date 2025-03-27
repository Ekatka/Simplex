import gym
from gym import spaces
import numpy as np
from my_simplex import _pivot_col, _pivot_row, _apply_pivot  # Your custom module with heuristics

class SimplexGymEnv(gym.Env):
    """
    A custom Gym environment where an RL agent chooses pivot strategies
    to solve linear programs via the simplex method.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, A, b, c, maxiter=1000, tol=1e-9):
        super(SimplexGymEnv, self).__init__()
        self.A = A
        self.b = b
        self.c = c
        self.m, self.n = A.shape
        self.maxiter = maxiter
        self.tol = tol

        self.action_space = spaces.Discrete(4)  # 4 strategies
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.m + 1, self.n + 1), dtype=np.float64
        )

        self.nit = 0
        self._initialize_tableau()



    def _initialize_tableau(self):
        self.basis = np.arange(self.m) + self.n
        self.T = np.hstack((self.A, np.eye(self.m), self.b[:, np.newaxis]))
        self.T = np.vstack((self.T, np.hstack((self.c, np.zeros(self.m + 1)))))

        # Pad tableau if needed
        if self.T.shape[1] != self.n + self.m + 1:
            raise ValueError("Invalid tableau shape.")

    def _get_obs(self):
        return self.T.copy()

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._initialize_tableau()
        self.nit = 0
        return self._get_obs(), {}

    def step(self, action):
        strategy_map = {
            0: 'bland',
            1: 'largest_coefficient',
            2: 'largest_increase',
            3: 'steepest_edge'
        }
        strategy = strategy_map[int(action)]
        done = False
        reward = -1  # Penalize per step only

        pivcol_found, pivcol = _pivot_col(self.T, tol=self.tol, strategy=strategy)
        if not pivcol_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        pivrow_found, pivrow = _pivot_row(self.T, self.basis, pivcol, phase=2, tol=self.tol)
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
        print("Iterations:", self.nit)

    def close(self):
        pass


def from_tableau(T, basis):
    return None