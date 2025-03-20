import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import my_simplex
from createMatrix import Matrix
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

    def _init_env(self, seed=None):

        perturbed_P = generate_perturbed_matrix(self.base_matrix, self.epsilon)
        A = np.hstack([-perturbed_P.T, np.ones((n, 1))])
        b = np.zeros(n)
        c = np.hstack([np.zeros(m), -1])
        one_row = np.hstack([np.ones(m), [0]])
        A = np.vstack([A, one_row])
        b = np.append(b, [1])
        self.env = my_simplex.SimplexGymEnv(A, b, c, maxiter=self.maxiter)
        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self, seed=None, **kwargs):

        self._init_env()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)



def evaluate_on_perturbed_matrices(base_matrix, num_tests=5, epsilon=0.1):
    pivot_map = {0: 'bland', 1: 'largest_coefficient', 2: 'largest_increase', 3: 'steepest_edge'}


    for i in range(num_tests):
        eval_P = generate_perturbed_matrix(base_matrix, epsilon)
        A = np.hstack([-eval_P.T, np.ones((n, 1))])
        b = np.zeros(n)
        c = np.hstack([np.zeros(m), -1])
        one_row = np.hstack([np.ones(m), [0]])
        A = np.vstack([A, one_row])
        b = np.append(b, [1])

        eval_env = my_simplex.SimplexGymEnv(A, b, c, maxiter=5000)
        obs, _ = eval_env.reset()
        done = False
        model = PPO.load("ppo_simplex_random_10x10")

        while not done:
            action, _ = model.predict(obs)
            chosen_method = pivot_map.get(int(action), 'bland')
            print(f"[Test {i + 1}] Chosen pivot method:", chosen_method)
            obs, reward, done, truncated, info = eval_env.step(action)

        pivot_steps_rl = eval_env.nit
        print(f"[RL Policy - Test {i + 1}] Pivot steps: {pivot_steps_rl}")
if __name__ == "__main__":
    matrix = Matrix()
    base_P = matrix.generateMatrix()
    m, n = matrix.returnSize()
    vec_env = make_vec_env(lambda: RandomMatrixEnv(base_P, epsilon=0.1), n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_simplex_random_10x10")
