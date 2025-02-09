
import numpy as np
import my_simplex

# minimize: -x - y
# subject to:
#    x + 2y <= 4
#    4x +  y <= 8
# and x, y >= 0

A = np.array([[1, 2],
              [4, 1]], dtype=float)
b = np.array([4, 8], dtype=float)
c = np.array([-1, -1], dtype=float)  # Minimizing -x - y => same as maximizing x+y

env = my_simplex.SimplexPivotEnv(A, b, c, maxiter=50)

obs = env.reset()
done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()  # 4 possible actions
    obs, reward, done, info = env.step(action)
    total_reward += reward

    env.render()

print(f"\nEpisode finished. total_reward = {total_reward}, info={info}")
