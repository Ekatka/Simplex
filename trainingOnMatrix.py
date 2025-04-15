import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import my_simplex
from my_simplex import prepare_phase2_tableau
from createMatrix import Matrix
# from gymEnv import SimplexGymEnv
from simplex_solver import solve_zero_sum, FirstPhasePivotingEnv
from simplex_solver import SecondPhasePivotingEnv
from simplex_solver import FirstPhasePivotingEnv, firstTosecond


def generate_perturbed_matrix(base_matrix, epsilon=0.1):
    rng = np.random.default_rng()
    noise = rng.uniform(-epsilon, epsilon, base_matrix.shape)
    return base_matrix + noise

def pad_observation(obs, target_shape):
    padded = np.zeros(target_shape, dtype=obs.dtype)
    current_shape = obs.shape
    padded[:current_shape[0], :current_shape[1]] = obs
    return padded


class RandomMatrixEnv(gym.Env):
    def __init__(self, base_matrix, epsilon=0.1, maxiter=5000):
        super().__init__()
        self.base_matrix = base_matrix
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.env = None
        self.phase = None
        self.av = None
        self._init_env()
        self.fixed_obs_shape = self.env.observation_space.shape
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.counter = 0
        self.nit = 0


    def _init_env(self, seed=None):
        self.nit = 0
        max_iter = 10

        for i in range(max_iter):
            perturbed_P = generate_perturbed_matrix(self.base_matrix, self.epsilon)
            res = solve_zero_sum(perturbed_P)
            if res is not None:
                T, basis, self.av = res
                self.phase = 1
                self.env = FirstPhasePivotingEnv(T, basis)
                if seed is not None:
                    self.env.reset(seed=seed)
                return

        print("too many unstable matricies")
        raise RuntimeError

    def reset(self, seed=None, **kwargs):
        self._init_env()
        obs, info = self.env.reset(seed=seed)
        obs = pad_observation(obs, self.fixed_obs_shape)
        return obs, info


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.nit+=1
        if self.phase == 1 and done:
            T_phase2, basis_phase2 = firstTosecond(self.env.T, self.env.basis, self.av)
            self.env = SecondPhasePivotingEnv(T_phase2, basis_phase2 )
            self.phase = 2
            obs, _ = self.env.reset()
            done = False
            self.env = SecondPhasePivotingEnv(T_phase2, basis_phase2)
            obs = pad_observation(obs, self.fixed_obs_shape)
            done = False
        if self.phase == 2:
            obs = pad_observation(obs, self.fixed_obs_shape)
        return obs, reward, done, truncated, info

        # return self.env.step(action)



if __name__ == "__main__":
    matrix = Matrix()
    base_P = matrix.generateMatrix()
    m, n = matrix.returnSize()
    epsilon = matrix.returnEpsilon()
    vec_env = make_vec_env(lambda: RandomMatrixEnv(base_P, epsilon), n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_simplex_random_100_000_steps_10x10")
