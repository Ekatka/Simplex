import numpy as np
# Training parameters
M = 5
N = 5
EPSILON = 0.1
TIMESTEPS = 100_000
MIN_VAL = -1
MAX_VAL = 1
BASE_MATRIX = np.array([[0,0,-1,-1,1],[1,0,-1,-1,1],[-1,-1,1,1,1],[1,1,1,1,-1],[-1,1,-1,-1,1]])
# BASE_MATRIX = np.array([[1,-1,0],[0,1,-1],[-1,0,1]])
# Training settings
N_ENVS = 4
# Model save path template
MODEL_NAME_TEMPLATE = "ppo_simplex_random_{steps}_matrix{m}x{n}_min{min}_max{max}_epsilon{eps}.zip"

# Max BFS depth
BFS_DEPTH = 10

# Action to pivot name map
PIVOT_MAP = {
    0: 'bland',
    1: 'largest_coefficient',
    2: 'largest_increase',
    3: 'steepest_edge'
}

