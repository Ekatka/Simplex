import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import my_simplex

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



'''

np.random.seed(50)
m, n = 10, 10
P = np.random.randint(-5, 6, size=(m, n))  # integer values in [-5, 5]
print("Payoff matrix P:\n", P)


A = np.hstack([-P.T, np.ones((n, 1))])
b = np.zeros(n)
c = np.hstack([np.zeros(m), -1])


one_row = np.hstack([np.ones(m), [0]])
A = np.vstack([A, one_row])
b = np.append(b, [1])

# env = my_simplex.SimplexGymEnv(A, b, c, maxiter=5000)
# vec_env = make_vec_env(lambda: env, n_envs=4)
# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=500000)
# model.save("ppo_simplex_10x10")


model = PPO.load("ppo_simplex_10x10")
eval_env = my_simplex.SimplexGymEnv(A, b, c, maxiter=5000)
obs, _ = eval_env.reset()
done = False
pivot_map = {0: 'bland', 1: 'largest_coefficient', 2: 'largest_increase'}
while not done:
    action, _ = model.predict(obs)
    chosen_method = pivot_map.get(int(action), 'bland')
    print("Chosen pivot method:", chosen_method)
    obs, reward, done, truncated, info = eval_env.step(action)
pivot_steps_rl = eval_env.nit
print(f"[RL Policy] Pivot steps: {pivot_steps_rl}")

def run_fixed_strategy(env, fixed_action):
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(fixed_action)
    return env.nit

bland_env = my_simplex.SimplexGymEnv(A, b, c, maxiter=5000)
pivot_steps_bland = run_fixed_strategy(bland_env, fixed_action=0)
pivot_steps_coefficient = run_fixed_strategy(bland_env, fixed_action=1)
pivot_steps_increase = run_fixed_strategy(bland_env, fixed_action=2)

print(f"Bland Pivot steps: {pivot_steps_bland}")
print(f"Coefficient Pivot steps: {pivot_steps_coefficient}")
print(f"Increase Pivot steps: {pivot_steps_increase}")
