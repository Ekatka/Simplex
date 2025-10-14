import numpy as np
# Training parameters
M = 100
N = 100
EPSILON = 0.001
TIMESTEPS = 50_000
MIN_VAL = -1000
MAX_VAL = 1000
LOAD_MODEL = False
PREFERRED_ACTION_ID = 0
INITIAL_BIAS = 3.0

# Training settings
N_ENVS = 4
# Model save path template
MODEL_NAME_TEMPLATE = "models/ppo_simplex_random_{steps}_matrix{m}x{n}_min{min}_max{max}_epsilon{eps}.zip"

# Max BFS depth
BFS_DEPTH = 10

# Action to pivot name map
PIVOT_MAP = {
    0: 'largest_coefficient',
    1: 'largest_increase', 
    2: 'steepest_edge',
    # 3: 'blands_rule'
}

NUM_PIVOT_STRATEGIES = len(PIVOT_MAP)
PIVOT_STRATEGY_NAMES = list(PIVOT_MAP.values())

# Advanced training features
USE_MACRO_STRATEGY = True  # Set to True to use MacroStrategyWrapper
USE_BIAS_ANNEALING = True  # Set to True to use LogitBiasAnnealCallback

