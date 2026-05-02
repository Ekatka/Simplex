import numpy as np

# Game mode: "matrix", "leduc", or "cubes"
GAME_MODE = "matrix"

# Leduc poker settings (only used when GAME_MODE = "leduc")
LEDUC_GAME = "leduc_poker(suit_isomorphism=true)"  # OpenSpiel game string
LEDUC_ALPHA = 100.0      # Dirichlet concentration: high alpha -> small perturbation around uniform
LEDUC_NUM_RANKS = 3     # Number of card ranks (J, Q, K)

# Cube settings (only used when GAME_MODE = "cubes")
# Each reset samples one of three pathological cube families (Klee-Minty,
# Jeroslow, Goldfarb-Sit) with the parameter drawn from the matching range.
CUBE_N = 10
# Tuned from the empirical sweep in docs/cubes_experiment.md — these
# parameter ranges produce distinguishable tableau patterns. Only the KM
# range actually hits 2^n pathology; JL and GS are currently KM-pathological
# variants with different tableau shapes (see cubes.py for caveats).
CUBE_EPS_RANGE = (0.6, 1.0)     # Klee-Minty tilt
CUBE_BETA_RANGE = (0.30, 0.45)  # Jeroslow objective decay
CUBE_DELTA_RANGE = (0.1, 0.3)   # Goldfarb-Sit row scaling

# Training parameters
M = 40
N = 40


TIMESTEPS = 20_000_000

# Checkpoint settings: save model every CHECKPOINT_FREQ steps after CHECKPOINT_START
CHECKPOINT_START = 1_000_000
CHECKPOINT_FREQ = 1_00_000
LOAD_MODEL = False
MATRIX_MODE = "toeplitz" # toeplitz, uniform

# UNIFORM MODE SETTINGS
MIN_VAL = -1
MAX_VAL = 1


# TOEPLITZ Matrices
TOEPLITZ_RHO = 0.95
TOEPLITZ_SIGNED = False
TOEPLITZ_ANTISYMMETRIC = False
TOEPLITZ_BAND = None

# EPSILON used in both
EPSILON = 0.001

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
    # 3: 'random_edge'
    # 3: 'blands_rule'
}

PIVOT_MAP_TEST = {
    0: 'largest_coefficient',
    1: 'largest_increase',
    2: 'steepest_edge',
    3: 'random_edge',
    4: 'blands_rule'
}

NUM_PIVOT_STRATEGIES = len(PIVOT_MAP)
NUM_PIVOT_STRATEGIES_TEST = len(PIVOT_MAP_TEST)
PIVOT_STRATEGY_NAMES = list(PIVOT_MAP_TEST.values())

# Advanced training features
USE_MACRO_STRATEGY = False  # Set to True to use MacroStrategyWrapper
USE_BIAS_ANNEALING = False # Set to True to use LogitBiasAnnealCallback
USE_INITIAL_ACTION_BIAS = False

# History tracking feature
USE_COMPACT_OBS = True  # Wrap env with CompactObsWrapper (31 size-independent features)
USE_BASELINE_REWARD = True  # Shape reward by difference from baseline (steepest_edge) iter count
BASELINE_REWARD_COEF = 2.0  # Multiplier for (baseline_nit - agent_nit) terminal bonus — amplified to push past "tie with steepest"
BASELINE_REWARD_WINS_ONLY = True  # If True, only reward strict wins vs baseline (no penalty for losing) — removes the "imitate steepest to avoid losses" equilibrium
USE_FULL_PIVOT = False  # Agent plays BOTH phases (True) or just phase 2 (False); ignored for cubes
USE_LR_DECAY = False  # Linear decay of learning rate (1e-4 -> 1e-5) over the run
ENT_COEF = 0.05  # PPO entropy coefficient (higher = more exploration / action diversity)
SWITCH_BONUS = 0.3  # Per-step reward for picking a different action than previous — encourages strategy diversity
USE_HISTORY_TRACKING = False  # Enable history tracking and printing when objective doesn't improve
HISTORY_SIZE = 20  # Number of last steps to keep in history
NO_IMPROVE_STEPS = 100  # Number of steps without improvement before printing history

# Single coordinate noise matrix feature
USE_SINGLE_COORDINATE_NOISE = False  # Set to True to use SingleCoordinateNoiseMatrix instead of Matrix
SINGLE_COORDINATE_NOISE_FLAG = False  # When USE_SINGLE_COORDINATE_NOISE=True, this controls the single_coordinate_noise flag

# Two-phase simplex: when True, build Phase 1 tableau and solve it (Bland's rule),
# then transition to Phase 2 for the RL agent. When False, construct
# the Phase 2 tableau directly (skipping Phase 1).
# NOTE: the two modes produce different tableau shapes, so models are NOT interchangeable.
USE_TWO_PHASE = True

# Weighted step penalty: scale the per-step penalty by the empirical cost of
# the chosen pivot rule (calibrate via benchmark_pivot_cost.py).
# When False, every step costs -1 regardless of strategy (current behavior).
USE_WEIGHTED_STEP_PENALTY = False
# Calibrated by benchmark_pivot_cost.py at 40x40 (matrix mode, phase 2),
# normalized so largest_coefficient = 1.0. Re-run the benchmark and update
# these if you change PIVOT_MAP or train at a different size/mode.
STEP_PENALTY_WEIGHTS = {
    'largest_coefficient': 1.00,
    'steepest_edge':       1.72,
    'largest_increase':    2.70,
    'random_edge':         0.86,
    'blands_rule':         0.37,
}

# Scale the per-step penalty linearly with tableau row count so the same
# weights generalize across LP sizes. Reference is set to a 40x40 matrix-mode
# phase-2 tableau (≈ 41 rows). Adjust if you train at a different size.
SCALE_PENALTY_BY_SIZE = False
REFERENCE_TABLEAU_ROWS = M + 1

