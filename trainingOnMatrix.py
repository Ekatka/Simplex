import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import my_simplex
from my_simplex import prepare_phase2_tableau
from createMatrix import Matrix
# from gymEnv import SimplexGymEnv
from simplex_solver import solve_zero_sum
from simplex_solver import PivotingEnv

'''
P = np.array([
    [0, -1,  1],  # Rock vs (Rock=0, Paper=-1, Scissors=1)
    [1,  0, -1],  # Paper vs ...
    [-1, 1,  0]   # Scissors
])

m, n = P.shape
A = np.hstack([-P.T, np.ones((n, 1))])
A = np.vstack([A, one_row])
b = np.zeros(n)# Now shape: (4, 4)
b = np.append(b, [1])
c = np.hstack([np.zeros(m), [-1]])

vyzkouseno na piskvorkach, vsechna pravidla davaji dohromady 5 kroku

seed 50 - horsi
seed 45 - lepsi >= coefficient 10
seed 46 - >= bland, increase 10
seed 43 > lepsi

najit si matici na ktery to funguje dobre a pak pricitat sum

'''


def generate_perturbed_matrix(base_matrix, epsilon=0.1):
    rng = np.random.default_rng()
    noise = rng.uniform(-epsilon, epsilon, base_matrix.shape)
    return base_matrix + noise


class RandomMatrixEnv(gym.Env):
    def __init__(self, base_matrix, epsilon=0.1, maxiter=5000):
        super().__init__()
        self.base_matrix = base_matrix
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.env = None
        self._init_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.counter = 0;

    def _init_env(self, seed=None):
        max_iter = 10
        for i in range(max_iter):
            perturbed_P = generate_perturbed_matrix(self.base_matrix, self.epsilon)
            res = solve_zero_sum(perturbed_P)
            if res is not None:
                T, basis = res
                self.env = PivotingEnv(T, basis)
                if seed is not None:
                    self.env.reset(seed=seed)
                return

        print("too many unstable matricies")
        raise RuntimeError

    def reset(self, seed=None, **kwargs):
        self._init_env()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


if __name__ == "__main__":
    matrix = Matrix()
    base_P = matrix.generateMatrix()
    m, n = matrix.returnSize()
    vec_env = make_vec_env(lambda: RandomMatrixEnv(base_P, epsilon=1), n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_simplex_random_10x10")
