import math
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch as th

# print number of episodes finished in each rollout
class EpisodeCounterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.completed_this_iter = 0

    def _on_rollout_start(self) -> None:
        self.completed_this_iter = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.completed_this_iter += 1
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("debug/episodes_finished_in_rollout", self.completed_this_iter)

# add bias to choosing actions
class LogitBiasAnnealCallback(BaseCallback):
    def __init__(self, preferred_id: int, initial_bias: float = 3.0, half_life: int = 500_000, verbose=0):
        super().__init__(verbose)
        self.preferred_id = int(preferred_id)
        self.initial_bias = float(initial_bias)
        self.half_life = int(half_life)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        factor = math.pow(0.5, t / max(1, self.half_life))
        with th.no_grad():
            bias = self.model.policy.action_net.bias
            bias[:] = 0.0
            bias[self.preferred_id] = self.initial_bias * factor
        return True

# was used to track history of actions and rewards
class HistoryTrackerCallback(BaseCallback):
    def __init__(self, history_size: int = 20, no_improve_steps: int = 100, improve_tol: float = 1e-12):
        super().__init__()
        self.history_size = int(history_size)
        self.no_improve_steps = int(no_improve_steps)
        self.improve_tol = float(improve_tol)

        self.history = []
        self._last_obj = None
        self._no_improve_counter = 0
        self._printed_this_episode = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])
        rewards = self.locals.get("rewards", [])

        if infos and len(infos) > 0:
            info = infos[0]

            if "episode" in info:
                self._last_obj = None
                self._no_improve_counter = 0
                self._printed_this_episode = False
                self.history = []
                return True

            obj = info.get("objective")
            action = actions[0] if actions is not None and len(actions) > 0 else None
            reward = rewards[0] if rewards is not None and len(rewards) > 0 else None
            strategy = info.get("strategy", "unknown")
            degenerate = info.get("degenerate", False)
            nit = info.get("nit", 0)

            step_info = {
                "nit": nit,
                "objective": obj,
                "action": action,
                "strategy": strategy,
                "reward": reward,
                "degenerate": degenerate
            }
            self.history.append(step_info)

            if len(self.history) > self.history_size:
                self.history.pop(0)

            if self._last_obj is not None and obj is not None:
                delta = self._last_obj - obj
                if delta > self.improve_tol:
                    self._no_improve_counter = 0
                else:
                    self._no_improve_counter += 1

            if obj is not None:
                self._last_obj = obj

            if self._no_improve_counter >= self.no_improve_steps and not self._printed_this_episode:
                self._print_history()
                self._printed_this_episode = True

        return True

    def _print_history(self):
        if not self.history:
            return

        print("\n" + "="*80)
        print(f"Objective hasn't improved for {self._no_improve_counter} steps")
        print(f"Last {len(self.history)} steps:")
        print("="*80)
        print(f"{'Step':<8} {'Objective':<15} {'Action':<10} {'Strategy':<20} {'Reward':<10} {'Degenerate':<12}")
        print("-"*80)

        from config import PIVOT_MAP

        for i, step in enumerate(self.history):
            nit = step.get("nit", "N/A")
            obj = step.get("objective", "N/A")
            action = step.get("action", "N/A")
            strategy = step.get("strategy", "N/A")
            reward = step.get("reward", "N/A")
            degenerate = step.get("degenerate", False)

            if isinstance(obj, (int, float)):
                obj_str = f"{obj:.8e}"
            else:
                obj_str = str(obj)

            if isinstance(action, (int, np.integer)):
                action_name = PIVOT_MAP.get(int(action), f"act_{action}")
            else:
                action_name = str(action)

            if isinstance(reward, (int, float)):
                reward_str = f"{reward:.4f}"
            else:
                reward_str = str(reward)

            degenerate_str = "Yes" if degenerate else "No"

            print(f"{nit:<8} {obj_str:<15} {action_name:<10} {strategy:<20} {reward_str:<10} {degenerate_str:<12}")
        print("="*80 + "\n")


