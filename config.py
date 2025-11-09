import numpy as np
# Training parameters
M = 50
N = 50

TIMESTEPS = 500_000
LOAD_MODEL = False
MATRIX_MODE = "uniform" # toeplitz, uniform

# UNIFORM MODE SETTINGS
MIN_VAL = -10
MAX_VAL = 10


# TOEPLITZ Matrices
TOEPLITZ_RHO = 0.80
TOEPLITZ_SIGNED = False
TOEPLITZ_ANTISYMMETRIC = False
TOEPLITZ_BAND = None

# EPSILON used in both
EPSILON = 0.1

PREFERRED_ACTION_ID = 2
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
USE_MACRO_STRATEGY = False  # Set to True to use MacroStrategyWrapper
USE_BIAS_ANNEALING = False # Set to True to use LogitBiasAnnealCallback
USE_INITIAL_ACTION_BIAS = False

# History tracking feature
USE_HISTORY_TRACKING = False  # Enable history tracking and printing when objective doesn't improve
HISTORY_SIZE = 20  # Number of last steps to keep in history
NO_IMPROVE_STEPS = 100  # Number of steps without improvement before printing history

