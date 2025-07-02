# Zero sum game solver using gym environment

## Installation

Clone the directory
```bash
git clone https://github.com/Ekatka/Simplex.git
cd Simplex
```

create the virtual environment
```bash
python -m venv venv
source venv/bin/activate   
```

install requirements
```bash
pip install -r requirements.txt
```

Modify the `config.py` file according to desired training and run `training_ppo_simplex.py`

## Documentation
Project consists of five main files:

- training_ppo_simplex.py - training agent
- simplex_solver.py - helper functions for training and testing
- matrix.py - defines Matrix class
- testing.py - tests agent compared to single strategies
- config.py - values configuration

### `training_ppo_simplex.py`

 Is responsible for training a reinforcement learning (RL) agent to solve zero-sum games using the two-phase simplex method within a custom gym environment.

It trains a PPO agent from Stable-Baselines3, where the agent learns to choose among four pivoting strategies at each step of the simplex algorithm:

- **Bland's Rule**
- **Largest Coefficient**
- **Largest Increase**
- **Steepest Edge**

Each training episode solves a randomly perturbed matrix game. The RL agent interacts with a custom environment that transitions between Phase 1 and Phase 2 of the simplex method.

---

#### Key Components

- **RandomMatrixEnv**:
  - A `gym.Env` that wraps around:
    - `FirstPhasePivotingEnv` - solving phase 1
    - `SecondPhasePivotingEnv` - solving phase 2
  - transitions between phases after Phase 1 completes
  - Pads observations to ensure shape compatibility across phases

- **Matrix** (from matrix.py):
  - Generates game matrices (base_P) and perturbed variants.
  - Encapsulates matrix parameters like size and epsilon perturbation.

- **Simplex utilities** (from simplex_solver.py):
  - `change_to_zero_sum`: Converts a payoff matrix to a zero-sum LP.
  - `first_to_second`: Converts Phase 1 tableau to Phase 2.
  - `FirstPhasePivotingEnv` and `SecondPhasePivotingEnv`: Gym environments for each simplex phase.

- **Configuration** (from config.py):
  - Includes values like `M`, `N`, `EPSILON`, `TIMESTEPS`, ...
  - Defines the `BASE_MATRIX` and pivot strategy mappings.

---

####  Functionality

- Sets up a Gym environment using `make_vec_env`
- Perturbs the base matrix to introduce variation across episodes
- Trains the PPO agent to learn which pivoting strategy to apply at each step
- Saves the trained model 

### `simplex_solver.py`

Implements the functionality of solving the linear program and provides gym environments for training.  

----

#### Key Components

- Gym Environments
Two environments represent Phase 1 and Phase 2 of the simplex algorithm. They support implemented pivoting heuristics.

  - **`FirstPhasePivotingEnv`**
    - Used to find a feasible solution by introducing artificial variables.

  - **`SecondPhasePivotingEnv`**
    - Used to optimize the objective once a feasible solution is found.
    - Returns the final game value from the tableau.

  Both environments:
  - Use `Discrete(4)` action space (representing 4 pivot strategies).
  - Penalize each step with a reward of `-1`.
  - Share a consistent observation space representing the current tableau.
- Pivoting 
  - `_pivot_col_heuristics` handles finding the right column according to chosen heuristic
  - `_apply_pivot` uses found pivot row and column to apply the pivoting step
- `change_to_zero_sum`: Converts a payoff matrix to a zero-sum LP.
- `first_to_second`: Converts Phase 1 tableau to Phase 2.

### `matrix.py`

Encapsulates logic related to creating, modifying, and perturbing game matrices

##### Key Methods

- generateMatrix() - creates a random matrix according to defined parameters, used in case if BASE_P is undefined in config
- generate_perturbed_matrix() - creates a noise matrix with interval from -EPSILON to EPSILON and then adds it to the original matrix

### `testing.py`

This script evaluates the performance of pivoting strategies, both fixed and learned

--

### Key Components

- **BFS Search (`find_optimal_pivot_sequence_bfs`)**
  - Exhaustively searches for the shortest sequence of pivot actions using breadth-first search.
  - Stops when a feasible solution is reached or `BFS_DEPTH` is exceeded.

- **Fixed Strategy Evaluation (`run_fixed_strategy`)**
  - Tests each of the 4 pivoting rules: Bland, Largest Coefficient, Largest Increase, Steepest Edge.
  - Outputs step count and final game value.

- **RL Policy Evaluation (`test_rl`)**
  - Loads a trained PPO model and steps through the environment using learned pivoting choices.
  - Reports actions taken, game value, and step count.

